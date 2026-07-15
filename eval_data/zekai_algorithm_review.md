# Efficiency + output-quality review of Zekai's workflow

Requested by Daniel. Everything below was measured offline against real data — no
API calls, nothing spent. Reproduce with the scripts in this repo.

**Framing first, because it matters:** Zekai's LLM-repair design is a *good idea*
and solves the right problem (language-agnostic segmentation, unlike a
weekday-regex). His prompt explicitly refuses to rely on Spanish weekdays, which
is the correct instinct. The issues below are concrete and fixable, and several
are already on his list (the Batch-API refactor fixes the biggest cost item).
Scope note: his rule-based script was written for the St-Augustine Spanish
volumes and never claimed corpus-wide coverage — the wide-corpus numbers below
are a *coverage* measurement Daniel asked for, not a verdict on work done in
scope.

---

## 1. The headline efficiency findings (`transcription_json_to_training_llm_repair.py`)

| # | Finding | Measured | Fix |
|---|---|---|---|
| E1 | **Every page is sent to the LLM ~2×.** `WINDOW=2, OVERLAP=1` → step=1, so each page appears in two windows. | 495-page volume → 494 calls, **988 page-sends = 2.00× per page** | Overlap exists to catch cross-page records; keep the *goal*, drop the *cost* — see R2. |
| E2 | **Few-shot prefix re-sent on every entry.** `extract.py` default `max_shots=1000`; no caching, no batching. | prefix ≈ **14,050 tokens per entry-call** | Batch + cache + fold (R1). This is the dominant cost, and the Batch-API refactor already targets it. |
| E3 | **Normalization is a separate LLM pass.** `LLM_REPAIR_RETURNS_NORMALIZED = False` by default. | **2× the per-entry calls** | Flip the flag he already wrote — the hook exists. |
| E4 | **Reasoning + sampling on a mechanical task.** `reasoning_effort="medium"`, `temperature=1`. | wasted reasoning tokens; run-to-run non-determinism | Segmentation is structural, not a reasoning task: `temperature=0`, drop reasoning effort. |
| E5 | **Model names don't resolve.** `gpt-5.4-nano`, `gpt-5.4mini`. | — | Pin real model ids before any batch run. |

## 2. The headline output-quality findings

| # | Finding | Measured | Fix |
|---|---|---|---|
| Q1 | **~17% duplicate entries.** The overlap dedup keys on `raw[:500]` only, so two windows that transcribe the same record with any early difference both survive. | **13 confirmed + 2 unconfirmed dupes / 88 entries** in his own outputs | Dedup on *sacrament principal + fuzzy full text* (implemented in our `qa.py`) — formulaic text alone isn't enough, since two different same-day baptisms are >0.9 similar. |
| Q2 | **Entry-count instability.** Identical pages, different runs/models → different entry counts. | reference **27** entries → variants produce **34–39** | Determinism (R2/E4). |
| Q3 | **The LLM's own `status` + `source_pages` are discarded.** The prompt asks for `incomplete_trailing` and `source_pages`; `normalize_repaired_entry_ids` keeps only `{id, raw}`. | paying for signal, then dropping it | Keep them — they're exactly the `partial` flag and cross-page provenance Daniel's gold format wants. |
| Q4 | **Window=2 can't assemble a record spanning >2 pages.** | structural limit | Deterministic stitch (R2) has no window limit. |
| Q5 | **Relationship extraction craters on the 5.4-family models.** | rels F1 **0.25–0.52** (5.4) vs **0.71–0.73** (4o); events look fine (0.95+) either way, which hides it | Choose the model on relationship F1 — it's SSDA's core product. See `model_agreement_0013_0023.md`. |
| Q6 | **`ethnicity` / `origin` / `phenotype` used inconsistently** (same value lands in different fields across runs). | agreement **0.00–0.17** on ethnicity | Tighten definitions in `instructions.json` — a schema fix, not a model fix. |

## 3. The rule-based script (`transcription_json_to_training_rule_based.py`)

This is the older/alternate path, but it's committed and it has a hard bug:

- **B1 — It cannot run.** `split_image_transcription_into_entries` calls
  `ENTRY_START_RE` (line 193), which is **never defined anywhere in the file**.
  Calling it raises `NameError: name 'ENTRY_START_RE' is not defined`. Verified
  by executing the function verbatim. → Define it or delete the dead path.
- **B2 — Silent whole-page fallback.** When no entry-start matches, it returns
  the **entire page as one "entry"** (no error, no flag). Because the pattern is
  Spanish-weekday-only, on the wider corpus this fires on **60,373 / 62,209 pages
  = 97.0%**, and on **180 of 232 volumes it fires on every page**. (Reconstructing
  the missing regex faithfully from his own comment to test the *approach*, not
  the typo.) On Daniel's new Portuguese gold: **his splitter returns 2 "entries"
  (one per page) where the gold has 10** — the first is a 1,569-char blob
  containing 5 real burials merged. Ours: 11 predicted, **9/10 exact text match**.
- **B3 — Silent data loss.** `should_keep_prelude` drops page-top continuation
  text under 15 words / 100 chars — i.e., the tail of a record continuing from
  the previous page is discarded.
- **B4 — `strip_trailing_marginalia`** drops trailing runs of ≤30-char lines,
  which can eat legitimate short signature lines (`O Vigr.o Joze …`).
- **B5 — Entry-id collisions.** `build_entry_id` uses only the image number
  (`0017-01`), dropping the volume, so ids collide across volumes.
- **B6 — `clean_text` only heals `-` line-breaks**, not `=`, which this corpus
  uses heavily (`nom`/`=bre`, `Mig=`/`=uel`).

## 4. Recommendations, in priority order

**R1 — Do the Batch-API refactor (already planned), and add two things with it.**
Batching ~10 entries per request + placing the static prefix first so it is
byte-identical across calls (prompt caching) + folding normalization into the
same call. Modelled on the real files: the whole corpus extracts for **~$52
(~$0.0008/image)**. Note: *don't* cut few-shot examples to save money — measured,
it saves a rounding error and costs accuracy; caching makes them nearly free.

**R2 — Replace LLM segmentation with deterministic segmentation + LLM fallback.**
Our rule-based segmenter handles ~94% of corpus pages confidently at **$0**, is
deterministic, has no window limit (so cross-page stitching is exact), and
structurally cannot produce Q1's duplicates. Route only the ~5–6% low-confidence
pages to the LLM. This deletes the entire repair pass — the pass that is both the
2× page-send cost (E1) *and* the source of the 17% duplicates (Q1).
Honest caveat: our segmenter's corpus-wide accuracy is **not yet certified**
(~36% exact / ~59% ±1 on a thin auto-check); it's strong on tested formats and
we're growing a gold set to measure it properly. So R2 is a strong recommendation,
not a proven-superior claim — the gold sheets will settle it.

**R3 — If the LLM repair stays:** fix the dedup key (Q1), keep `status` +
`source_pages` (Q3), `temperature=0` and no reasoning effort (E4), pin real
models (E5), and flip `LLM_REPAIR_RETURNS_NORMALIZED` (E3).

**R4 — Fix or remove the rule-based path** (B1) so nobody runs dead code.

**R5 — Independent of algorithm:** re-transcribe the **1,281 corpus pages** whose
"transcription" is actually an Archivault error string, and tighten the
ethnicity/origin/phenotype definitions (Q6) before any large extraction run.

## 5. Reproduce

```bash
python run_segment.py <volume>.json --structural        # our segmenter + structural check
python run_qa.py Sample_output/Generated_0024_0034_4o_prompt_V2.json   # the duplicate finding
python run_cost.py --target 0.01                        # the cost model + lever waterfall
python -m pytest tests -q                               # 72 offline tests
```
