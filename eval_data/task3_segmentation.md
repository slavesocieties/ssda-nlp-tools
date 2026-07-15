# Task 3: Archivault → entry segmentation — deterministic, ~$0/image

**Bottom line:** a rule-based state-machine segmenter (`ssda_nlp_tools/segment.py`)
reproduces Daniel's paired examples essentially perfectly, matches the register's
own margin numbering on 100% of pages of the Spanish test volume, and segments
**92.4% of a 17,373-page / 69-volume corpus sample confidently at $0.00/image**.
The remaining 7.6% of pages route to an LLM fallback (or human review), keeping
the whole segmentation stage in the **$0.0000x/image** range — versus an
LLM-only approach (Zekai's window repair), which costs a full model pass per
page *and* introduced the ~17% duplicate-entry problem.

## Verified results

| benchmark | result |
|---|---|
| Daniel's gold pair 1 (Pt baptisms, .md input, trailing partial) | **4/4 entries**, sims ≥0.99 space-insensitive, partial flags exact |
| Daniel's gold pair 2 (Pt burials, interleaved margins) | **4/4 entries**, sims ≥0.92, flags exact |
| Volume 239746 chunk 0013-0023 vs 4o reference | **recall 1.00 / precision 0.955**, coverage 1.00 |
| Structural check vs the register's own margin numbers | **32/32 pages agree (100%)** |
| Corpus sweep (69 volumes, 17,373 pages) | 54,717 entries (3.15/page), **7.6% pages flagged for fallback**, 0 crashes |

Chunks 0024-0044 score lower against the 4o *reference* (F1 ~0.8), but every
discrepancy was traced by hand to reference defects: the 4o output contains
window-overlap duplicates, entries split mid-sentence at page turns (which this
segmenter stitches correctly, per the gold spec), and inconsistently removed
margin text. Coverage (all reference content present in predictions) is 0.95–1.0.
The margin-number check — objective, LLM-free — is 100%.

## How it works (multilingual by construction)

Line-classifying **state machine** (JUNK / OPENER / CLOSER / SIGNATURE / TEXT),
not a regex split:

* **Openers**: Spanish + Portuguese date formulas ("Lunes, dia veinte y uno de
  Octubre de Mil…", "Aos vinte e quatro de Dezembro de mil…", "Em 3 de Agosto…"),
  weekday-led starts, b/v scribal variants ("Juebes", "Savado"), months broken
  across the line wrap ("…de Noviem / bre…") via two-line lookahead — but a
  margin name directly above an opener can never become the entry start.
* **In-entry date mentions don't split**: "nascida a vinte e dous de Novembro"
  (a birth date inside a baptism) stays in the entry; mid-entry splits need a
  strong opener AND a closed/long current entry.
* **Closers/signatures**: "…y lo firmé en dicho dia, mes, y Año=", "fiz este
  assento, que assigno", "O Vigr.o …", priest-name lines.
* **Junk**: page headers ("pag. 40", "1793."), margin number blocks ("47.."),
  standalone month markers ("Julho nada"), catchwords/reclamos ("=ma", "=ceno").
* **Margin handling per the gold pairs**: margin-name prefixes stripped from
  opener lines; interleaved margin words inside body lines kept.
* **Line re-flow**: "-" and "=" continuation marks healed ("Setecien/-tos",
  "nom/=bre", "Mig=/=uel"), while stop-"=" before signatures is preserved.
* **Cross-page stitching**: a page-top continuation attaches to the previous
  page's `partial` entry (21 stitches across the test volume); a trailing entry
  without its closer is emitted `"partial": true` exactly like the gold pairs.
* **Page typing**: near-empty pages = covers (fine, skip); pipe-table /
  folio-reference pages = **indexes** (skip — ~5,600 index rows that would have
  polluted the entry stream are now excluded); text-rich pages with no anchored
  entries and no attachment = **low-confidence → LLM fallback**.

Output matches the paired examples: `{"image", "entries": [{"id": "<stem>-NN",
"text", "partial"}]}`, plus volume mode with `source_images` provenance.

## Cost

* Deterministic pass: **$0.00** for 92.4% of pages.
* Fallback for the 7.6%: with the batched, cache-ordered prompt from
  `batch_extract.py` at gpt-4o-mini rates, ≈$0.0005/page → **≈$0.00004 per
  corpus image amortized**. Segmentation is a rounding error in the budget.
* This *replaces* the LLM window-repair pass entirely for confident pages —
  the pass that both cost a model call per page and created the duplicate
  entries. Combined with the earlier work, the full post-transcription pipeline
  stays at ~$0.001–0.002/image, well under the $0.01 target ($0.05 hard cap).

## What the flagged 7.6% actually is

Sampled by hand: archivist front-matter in English (1972 notes), index volumes,
administrative petitions (volume 266286 — not a sacramental register at all).
Routing these away from entry segmentation is correct behavior, and several
would otherwise silently pollute the database.

## Reproduce

```bash
python run_segment.py "Text data/SSDA_0013_0023_Gemini_V2.json" --structural --eval Sample_output/Generated_0013_0023_4o_prompt_V2.json
python run_segment.py <corpus>/239746*.json --out segmented.json     # volume mode
python -m pytest tests -q                                            # 65 tests
```

## FULL-CORPUS RUN (the complete Archivault transcription set, 2026-07-14)

Ronak supplied the complete corpus as a zip (232 volumes with transcriptions).
Full segmentation run — outputs in `out_corpus/<volume>.segmented.json` +
`out_corpus/corpus_summary.json`:

| metric | value |
|---|---|
| volumes processed | **232, zero crashes** |
| pages | 62,209 (in 33 s ≈ 1,900 pages/sec) |
| entries produced | **175,917** (2.83/page) |
| cross-page entries stitched | 21,634 |
| pages routed to LLM fallback | 3,729 (6.0%) |
| **pages containing embedded Archivault API failures** | **1,281 (2.1%)** |

**Upstream finding:** 1,281 pages carry verbatim failure strings inside their
"transcriptions" — `[transcription failed: max retries reached]`,
`finish reason: FinishReason.STOP` — i.e., ~2% of the corpus was never actually
transcribed and must be RE-SUBMITTED to Archivault. The segmenter tags these
`page_type: "error"` and excludes them; each volume's `error_pages` list is the
re-transcription worklist.

Closer formulas were extended data-driven from the corpus (Portuguese
"que assignei", "mandei fazer este assento"; Spanish "para que conste",
"…obligaciones y parentesco espiritual") — partial-flag noise fell from 43% to
15.1%, the remainder being genuinely split entries and rarer regional formulas.

Caveats: gold is small (2 pages, 8 entries) — the corpus sweep measures
*confidence and structure*, not text accuracy, beyond the evaluated volume.
The obvious next step is `run_goldprep.py`-style correction of a few segmented
volumes to grow segmentation gold cheaply.
