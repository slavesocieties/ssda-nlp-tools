# Using ssda-nlp-tools: from transcriptions to a QA'd dataset

This guide walks through one full run: segment a volume, stage the extraction
requests, **send them and collect the results**, then run QA, identity
resolution and the social graph. The README describes what each stage does.
This file lists the exact commands. Every command was dry-run against the
sample data in `Text data/`.

Only step 3 costs money. Every paid command is a dry run unless you add
`--confirm`, so run it once without `--confirm` and read the output first.

## 0. Which runner sends the requests?

There are two ways to send extraction requests. Pick one:

| | `run_live_test.py` | `run_luna_production.py` |
|---|---|---|
| Use it for | a small spot check: real cost per image, eyeball quality | real extraction of whole volumes |
| How it sends | synchronous chat calls, answers in seconds | OpenAI **Batch API** (50% cheaper, up to 24 h turnaround) |
| Size | `--max-batches N` requests (default 3) | up to `--take N` requests per job (default 50) |
| Spend guard | `--max-usd` for this run only (default $0.50) | a cumulative ledger with a hard cap (`--cap-usd`, default $200) |
| Validation | parses the output, reports missing entries | checks IDs, stop reason, JSON, schema and usage; writes an `accepted` file |
| Output | one `live_test_results.json` | receipt, raw output, accepted output and validation report per job |
| Dependencies | `pip install openai` | standard library only |

`run_model_bakeoff.py` compares *providers* on held-out entries (README,
"Optional live model calibration"). `run_corpus_prompts.py --expand` only
**writes** a Batch-API upload file. If you upload that file by hand you get no
ledger, no validation and no `accepted` file, so use `run_luna_production.py`
instead.

## 1. Segment a volume ($0)

The staging step finds files named `<volume>.segmented.json` in a corpus
directory. It takes the **volume ID from the file name**, and that ID is
stamped into every request's `custom_id`. Name the file after the real volume:

```bash
python run_segment.py "Text data/SSDA_0013_0023_Gemini_V2.json" --structural --out production/corpus/239746.segmented.json
```

For the six Drive volumes, `python run_production.py` builds
`production/corpus/` for you, using the routing manifests.

## 2. Stage the extraction requests ($0)

```bash
python run_corpus_prompts.py --corpus production/corpus --outdir production/batches --model gpt-5.6-luna --reasoning low --volumes 239746
```

This writes `production/batches/239746.batches.jsonl` and prints the projected
cost. `manifest.json` records the number of calls per volume.

> **Always pass `--model`.** The default is `claude-haiku-4.5`, but both
> runners send to OpenAI, so requests staged with the default model will fail
> at send time. Use an OpenAI model (`gpt-5.6-luna` is the evaluated choice).
> `run_luna_production.py` bills at Luna Batch prices regardless of model.

## 3. Send the requests and collect the results (PAID)

Both runners read the key from the environment only, and never print or store
it.

```powershell
$env:OPENAI_API_KEY = "sk-..."          # PowerShell
```
(bash: `export OPENAI_API_KEY=sk-...`)

### Path A: quick live test (a few requests, answers in seconds)

```bash
python run_live_test.py production/batches/239746.batches.jsonl --max-batches 3
```

The dry run shows the batch count, entries, images and projected cost. To send:

```bash
python run_live_test.py production/batches/239746.batches.jsonl --max-batches 3 --confirm --out live_239746.json
```

It prints actual cost per entry and per image next to the projection, then
writes `live_239746.json`. That file stores its records under `"entries"`, so
it goes straight into step 4.

### Path B: production Batch run (whole volumes)

Give each volume its own output directory and share **one** ledger, so every
job counts against the same spend cap.

The default cap is **$200** across all jobs combined. A ledger created before
that change still records the old $20 cap, and the runner refuses to run until
the two agree. Add `--raise-cap` once: the dry run shows `would raise ledger
cap $20 -> $200`, and the first `--confirm` run writes the new cap into the
ledger with a `cap_history` entry. After that, drop the flag.

```bash
# B1. dry run: shows requests, hard reservation, and remaining headroom under the cap
python run_luna_production.py production/batches/239746.batches.jsonl --outdir production/luna_239746 --ledger-path production/luna_live/spend_ledger.json --take 50

# B2. submit
python run_luna_production.py production/batches/239746.batches.jsonl --outdir production/luna_239746 --ledger-path production/luna_live/spend_ledger.json --take 50 --confirm
```

B2 prints `SUBMITTED batch_abc123...: N requests, reservation $X`. It also
writes `production/luna_239746/batch_abc123....receipt.json`, which holds the
job ID if you lose track of it.

If a volume has more calls than `--take`, repeat B2. The ledger records which
`custom_id`s were already sent, so each run submits the next unsent chunk. When
nothing is left it prints "No unsent compact requests remain".

**Collect** after the job finishes (usually within minutes to hours, 24 h at
most). Use the same batch file, `--outdir` and `--ledger-path` as when you
submitted:

```bash
# B3. poll/download/validate; safe to repeat while the job is still running
python run_luna_production.py production/batches/239746.batches.jsonl --outdir production/luna_239746 --ledger-path production/luna_live/spend_ledger.json --poll batch_abc123... --confirm
```

- `provider job ...: in_progress; no settlement performed`: the job is still
  running. Try again later. Polling is free.
- `VALIDATED batch_...`: done. The reservation is replaced by the cost the
  provider reported.
- `INVALID: ...` (exit code 2): some requests failed validation. Read
  `<job>.validation.json` for the reasons. Requests that passed are still
  written to the accepted file. To record the job's real cost in the ledger
  and keep the failed requests marked for repair, rerun B3 with
  `--settle-invalid`.

B3 writes three files to `production/luna_239746/`:

| file | contents |
|---|---|
| `<job>.output.jsonl` | raw provider output (audit only; never used for delivery) |
| `<job>.accepted.jsonl` | only the requests that passed every check. **This file feeds step 4.** |
| `<job>.validation.json` | the verdict, errors, token usage and confirmed cost |

**Materialize** the accepted rows into one volume file. The volume file takes
the original segmented text, image names and `partial` flags from the corpus,
and only `normalized` and `data` from the model. Run this from the repo root
with a **relative** glob, because absolute patterns fail:

```bash
python materialize_luna_results.py production/corpus/239746.segmented.json "production/luna_239746/*.accepted.jsonl" --out production/luna_239746/239746.materialized.json
```

The command refuses if any corpus entry is missing from the output. Add
`--allow-incomplete` to materialize a partial volume, for example while some
jobs are still running. Coverage is then reported in the file.

> **Re-extracting a volume that already has results** (e.g. after a prompt
> change): add `--run-id v2` and use a new `--outdir`. Keep the same
> `--ledger-path`. Without a new run ID, the ledger treats those requests as
> already sent. See RUNBOOK.md §2 for why both rules exist.

> **Several volumes at once:** `assemble_corpus.py` replaces the materialize
> step and step 4 for a whole directory of results. It assembles every volume
> that has a `<vol>.segmented.json` in `--corpus`, or only the ones you list
> with `--volumes`, and adds a cross-volume graph:
>
> ```bash
> python assemble_corpus.py --live production/luna_239746 --corpus production/corpus --volumes 239746
> ```
>
> Unlike `materialize_luna_results.py`, it reports missing records instead of
> refusing. It also lists any accepted rows that match none of the volumes in
> `CORPUS_SUMMARY.json`. Without `--volumes` it takes every volume in
> `--corpus`, and a volume with no results in `--live` counts as fully missing.
> So once new volumes sit in `production/corpus/`, assemble the delivered run
> with its five volumes named:
> `python assemble_corpus.py --volumes 176899 201991 29597 375062 701054`.

## 4. QA, identities, graph, human review ($0)

```bash
python run_pipeline.py production/luna_239746/239746.materialized.json --tag 239746 --outdir out_239746
```

This writes `qa_report.json`, `resolved.json`, `person_index.json`,
`network.graphml`, `nodes.csv`, `edges.csv`, `review.html` and `summary.txt`
to `out_239746/`. To link people across volumes, pass several materialized
files in one call.

To review the borderline identity merges:

1. Open `out_239746/review.html` in a browser.
2. Press `s` (same person), `d` (different) or `u` (unsure) for each pair.
3. Click "Download decisions.json".
4. Fold the decisions back in:

```bash
python run_review.py apply production/luna_239746/239746.materialized.json decisions.json --graphml out_239746/network.graphml --resolved out_239746/resolved.json --tag 239746
```

Other stand-alone checks: `run_qa.py` (QA only), `run_eval.py` (P/R/F1 against
gold, or agreement between two models) and `run_person_review.py`.

## Cheat sheet

| step | command | cost |
|---|---|---|
| segment | `run_segment.py IN.json --out production/corpus/<vol>.segmented.json` | $0 |
| stage | `run_corpus_prompts.py --corpus production/corpus --outdir production/batches --model gpt-5.6-luna --volumes <vol>` | $0 |
| spot check | `run_live_test.py production/batches/<vol>.batches.jsonl [--confirm]` | ~cents |
| submit | `run_luna_production.py <batchfile> --outdir production/luna_<vol> --ledger-path production/luna_live/spend_ledger.json [--confirm]` | PAID |
| collect | same command `+ --poll <job_id> --confirm` | $0 |
| materialize | `materialize_luna_results.py production/corpus/<vol>.segmented.json "production/luna_<vol>/*.accepted.jsonl" --out ...` | $0 |
| QA + graph | `run_pipeline.py <materialized>.json --tag <vol> --outdir out_<vol>` | $0 |

## Troubleshooting

| message | fix |
|---|---|
| `a non-default --outdir requires --ledger-path` | Add `--ledger-path production/luna_live/spend_ledger.json`. This is intentional: a new directory must not start a new budget. |
| `ledger cap $20 differs from requested $200 (pass --raise-cap ...)` | The ledger was created under the old $20 default. Add `--raise-cap` once. The dry run shows the change, and `--confirm` writes it to the ledger along with a `cap_history` entry. A cap is never lowered. |
| `REFUSING: reservation would exceed the cumulative hard cap` | Lower `--take`, or approve a higher cap. |
| `job is not present in local ledger` | You polled with a different `--ledger-path` or `--outdir` than you submitted with. |
| `NotImplementedError: Non-relative patterns are unsupported` | The materialize glob must be relative. Run from the repo root. |
| `run_pipeline.py` reports `(0 entries)` | The input has no `"entries"` key. Pass a materialized file, `run_live_test.py` output, or an `assemble_corpus.py` result. |
| `OPENAI_API_KEY is not set` | Set it in the **same** shell session that runs the command. |
