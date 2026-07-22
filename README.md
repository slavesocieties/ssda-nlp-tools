# ssda-nlp-tools

Offline tooling for the SSDA post-transcription NLP stage: turn Archivault
page transcriptions into **segmented sacramental entries**, **structured
people/events data**, **resolved cross-entry identities**, and the
**person/relationship social graph** — with evaluation, QA, cost modeling, and
a human review loop at every step.

**Design rule: paid calls are always deliberate.** The pipeline tools work
deterministically ($0) or *prepare and price* LLM work. The opt-in live
runners (`run_live_test.py` and `run_model_bakeoff.py`) are dry-run by
default and need an explicit `--confirm`. The full test suite (105 tests) runs
offline in under a second.

## The pipeline

```
Archivault transcriptions (per image)
  │  run_segment.py          deterministic entry segmentation      $0
  ▼
segmented entries (id / text / partial, cross-page stitched)
  │  run_corpus_prompts.py   priced, ready-to-send extraction batches
  ▼                          (OpenAI Batch-API JSONL; prepared, not submitted)
extracted people/events per entry
  │  run_qa.py               duplicates, chronology, dangling refs, drift
  │  run_eval.py             P/R/F1 vs gold or model-vs-model agreement
  │  run_disambiguate.py     cross-entry identity clustering + review queue
  │  run_review.py           human review page (s/d/u) -> must/cannot constraints
  │  run_link.py             link people ACROSS chunks/volumes
  ▼
run_network.py / run_pipeline.py -> person registry + GraphML social graph
```

## Headline results (all reproducible offline)

* **Task 3 segmentation**: the two paired gold examples reproduced 8/8
  entries (partial flags exact); **100% agreement with the register's own
  margin numbering** (32/32 pages, volume 239746); full corpus run = **232
  volumes / 62,209 pages → 175,917 entries in ~30 s, zero crashes**; 6% of
  pages routed to LLM fallback; **1,281 pages found to contain verbatim
  Archivault API-failure text** (re-transcription worklist per volume).
  Details: `eval_data/task3_segmentation.md`.
* **Cost**: extraction for the *entire* available corpus projects to
  **~$52 via the Batch API** (~$0.0008/image) with the full-quality 15-shot
  prompt and normalization folded in — far under the $0.01/image target.
  Levers and math: `eval_data/cost_to_penny.md`.
* **Identity resolution**: domain-guarded scorer (a person is baptized once;
  enslaver/spouse/parents are per-life constants; estate surnames don't imply
  identity; bare names need context) verified on real volume 239746 —
  priests O'Reilly ×45 / Hassett ×39 correctly linked across 88 pages while
  same-named infants and enslaved people stay distinct.
  Model bake-off: `eval_data/model_agreement_0013_0023.md`.

## Quick start

```bash
python -m pytest tests -q                          # 105 tests, no network, <1s

# segment one volume (Archivault JSON or the paired-example .md format)
python run_segment.py path/to/VOLUME.json --structural --out segmented.json

# stage priced extraction batches from a directory of segmented volumes
python run_corpus_prompts.py --corpus out_corpus --outdir out_batches
python run_corpus_prompts.py --expand out_batches/<vol>.batches.jsonl   # Batch-API file

# one-command QA + identities + graph + review page for extracted volumes
python run_pipeline.py EXTRACTED.json --tag VOL --outdir out_vol
```

Requirements: Python 3.10+ standard library only (pytest to run tests;
`tiktoken` optional for exact token counts). `ssda_nlp_tools/README.md` has the
full module-by-module documentation.

### Optional live model calibration

`run_model_bakeoff.py` compares the configured providers on held-out entries.
It never reads a key file: set the relevant provider key in the process
environment, inspect the default dry run, then add `--confirm` only when a
live test is approved.

```bash
python run_model_bakeoff.py --models gpt-5.4-mini --max-batches 1
python run_model_bakeoff.py --models gpt-5.4-mini --max-batches 1 --confirm
```

Before a live request it reserves each model's conservative worst-case amount
in `model_bakeoff_spend_ledger.json`. A lost network response remains reserved,
so resuming cannot silently reuse the same budget. The ledger and all
live-provider outputs are gitignored.

## Repo layout

```
ssda_nlp_tools/     the package (segmentation, eval, QA, identity, network, cost)
run_*.py            one CLI per pipeline stage (see ssda_nlp_tools/README.md)
tests/              105 offline tests; tests/fixtures/ = the paired gold examples
eval_data/          measured reports: segmentation, cost, model bake-off
Text data/, Sample_output/, Reduction_test/, training_data.json, instructions.json,
extract.py, normalize.py, utility.py, transcription_json_to_training_*.py
                    upstream project files + sample data these tools build on,
                    from github.com/Suzreal/SSDA_New_Workflow_Update (Zekai) and
                    github.com/slavesocieties/openai — kept verbatim so the
                    tests and cost measurements run against the real thing
out_corpus/, out_batches/   (gitignored) regenerable corpus outputs
```

## Honest limitations

* Segmentation gold is 2 pages / 8 entries + one fully-verified volume; corpus
  numbers measure structure and confidence, not per-entry text accuracy.
  `run_goldprep.py` makes growing gold cheap.
* ~15% of corpus entries carry a truthful `partial` flag (page-boundary or
  unrecognized regional closing formulas) — they are included and tagged, never
  dropped.
* The 6% low-confidence pages and 1,281 failed-transcription pages need,
  respectively, an LLM pass and Archivault re-submission — both are staged as
  worklists, neither is executed by this repo.
