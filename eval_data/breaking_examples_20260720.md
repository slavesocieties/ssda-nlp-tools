# Breaking-fix examples (2026-07-20) — segmentation result & diagnosis

Five paired gold examples from Daniel (`_sample.json` input → `_sample_output.json`
gold). Three seen before (65858, 420550, 740018), two new (**260950, 544367**).

## Result: perfect recall, two over-splits

| stem | gold | pred | records matched | note |
|---|---|---|---|---|
| 65858 | 10 | **11** | 10/10 | +1 trailing partial (real 21-May record gold omits) |
| 260950 | 13 | 13 | 13/13 | ✓ exact (Portuguese, 1910) |
| 420550 | 8 | 8 | 8/8 | ✓ exact |
| 544367 | 5 | **6** | 5/5 | +1 re-transcription duplicate across page break |
| 740018 | 11 | 11 | 11/11 | ✓ exact |

**Every gold record boundary is found (47/47 recall).** The two misses are
*precision* over-splits, both on 2-page inputs, both from the same upstream
cause: Archivault re-transcribes text at the page boundary.

- **544367**: page 0107 ends mid-record (partial, no signature); page 0108
  re-opens with that record's own opener (`En la Parroquia… a los cuatro días…
  de Octubre… 1895`), plus margin names interleaved into the body
  (`cuatro`→`cuaJuantro`). The segmenter reads it as a fresh record → duplicate.
- **65858**: the extra record is `Aos vinte e hum dias` (21 May) — a *different*
  date from the preceding `Aos vinte dias` (20 May). It is a **genuine new
  record** that runs off the last provided page (`partial:True`). Gold omits it;
  the segmenter is arguably *more* complete here.

## Why this is NOT auto-fixed in the segmenter

Consecutive sacramental records are near-identical in form, so content
heuristics cannot separate a re-transcription duplicate from two real records:

| candidate rule | 544367 (want flagged) | 65858 (must NOT) | 420550 (must NOT) |
|---|---|---|---|
| opener prefix-similarity | 0.94 | **0.93** | 0.65 |
| same parsed date as prev | unparseable (`cuaJuantro`) | diff date | **false-flags** |

Every rule aggressive enough to drop 544367's duplicate also deletes the real
21-May record in 65858 or breaks the already-correct 420550. On 62k pages that
means silently deleting real baptisms/burials. The segmenter therefore keeps
both and flags `partial` — the **safe** failure mode (over-inclusion, never
loss). Correct home for the fix, in order of preference:

1. **Upstream** — Archivault should not duplicate opener text across page
   boundaries (same family as the known 1,281 API-failure pages).
2. **QA / dedup** (`run_qa.py` already does "duplicates") — once records are
   completed by LLM extraction, a full-record duplicate like 544367's is
   trivially separable from 65858's distinct record; the partial fragment's
   scarce text is the problem, not the completed entry.
3. **Human review** (`run_review.py`) for the residue.

## Open convention question for Daniel

Should a trailing partial that runs off the **last provided page** (65858's
21-May record) be **kept + flagged** (current behavior, matches the 2026-07-16
"never drop partials" decision) or **dropped** (this gold's convention)? The two
new examples pull in opposite directions and only Daniel can set the rule.
