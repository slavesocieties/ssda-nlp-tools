# SSDA NLP pipeline — project status & how it works
*Single source of truth. Last updated 2026-07-22 (commit `9b0b24f`). Supersedes
the 2026-07-15 and 2026-07-20 supervisor drafts.*

This document covers the whole Task-3 pipeline — what it does, exactly how each
stage works, the current state (done / staged / open), the measured evidence,
and the commands to run it. It is written to be read start-to-finish by a
supervisor or a new contributor. Detailed per-topic reports live in `eval_data/`
and are linked inline.

---

## 1. What the pipeline is for

Turn Archivault page **transcriptions** of colonial sacramental registers into:

1. **Segmented entries** — one record per baptism / marriage / burial.
2. **Structured data** — people, events, and typed relationships per entry.
3. **Resolved identities** — the same person unified across entries and volumes.
4. **A social graph** — kinship + enslavement edges over those people.

Design rule enforced everywhere: **deterministic work is free and always runs;
the LLM is a capped, explicitly-approved step, never automatic.** No script
spends money without a dry-run and `--confirm`.

---

## 2. How it works — stage by stage

```
Archivault JSON (per volume)
  │  run_route_volume.py     classify source + per-page routing      $0
  ▼
routing manifest (deterministic-sacramental | fallback | retranscribe | index | admin)
  │  run_production.py       segment + tag records by disposition    $0
  ▼
production records  {id, text (faithful), images[], partial}         5,235 records
  │  run_corpus_prompts.py   price + stage ready-to-send batches     $0 (prepares)
  ▼
staged Luna batches  ── (dry-run → APPROVE → --confirm) ──► extracted people/events   ~$15 Batch API
  │  run_pipeline.py         QA → identity → link → graph            $0
  ▼
qa_report · resolved.json · person_index · network.graphml · review.html
```

### 2.1 Routing — `run_route_volume.py` (`ssda_nlp_tools/routing.py`)
Reads a volume export and classifies each page deterministically (title cues,
genre, page shape, index/error detection, segmentation confidence). Ambiguous
input routes to QA, never a guess. Output: a per-page manifest. Administrative
material (e.g. volume 3952, cofradía dossiers) is separated from sacramental
registers and never fed to the entry splitter.

### 2.2 Segmentation — `run_segment.py` (`ssda_nlp_tools/segment.py`)
A deterministic state machine splits a page's transcription into entries by
recognizing opening formulas ("En la Parroquia…", "Aos … dias do mês…"),
closing formulas, and signatures, in Spanish and Portuguese. It:
- **stitches records across page boundaries** and lists `source_images`;
- keeps **partial** (page-truncated) records and flags them — never drops them;
- strips margin-column names spliced into the body and heals line-wrapped words;
- assigns each record an id `<first-image-stem>-NN`.
Cost: **$0**. Pages below a confidence threshold are the only ones routed to the
LLM fallback.

### 2.3 Production assembly — `run_production.py`
Segments every sacramental volume and tags each record by the **worst** routing
disposition of the pages it spans, so nothing paid or broken leaks into the free
output. Emits production record sets + the corpus files the next step prices.

### 2.4 Extraction (the only paid step) — `run_corpus_prompts.py` → Batch API
Builds cache-ordered messages (static 15-shot prefix identical across every
call → prompt-cache hits; only the batch tail varies), folds normalization into
the same call, and returns per-entry `{normalized text, people, events}`. Keeps
**both** faithful and normalized text. `run_corpus_prompts.py` *prepares and
prices* the batches with **zero API calls**; sending is a separate, approved
step. Recommended model: **GPT-5.6 Luna** (§4).

### 2.5 Downstream — `run_pipeline.py` (QA, identity, graph)
Runs on the extracted output, all deterministic ($0):
- **QA** (`qa.py`) — near-duplicate entries (guarded by the sacrament
  *principal*, so two formulaic-but-distinct baptisms aren't merged), chronology,
  dangling references, event-shape rules, vocabulary drift.
- **Identity resolution** (`disambiguate.py`) — merges person-mentions into
  identities; domain-guarded (a person is baptized once; enslaver/parents are
  per-life constants; bare names need context).
- **Cross-chunk linking** (`link.py`) — unifies people across volumes.
- **Social graph** (`network.py`) — person registry + GraphML.
- **Review UI** (`review_html.py`) — a page to decide borderline merges.

---

## 3. Current state — done / staged / open

### ✅ Done and validated
- **Segmentation**: **47/47 record recall** across Daniel's 5 gold examples
  (Portuguese 1817 & 1910, 18th-c Spanish, Colombian 1895).
  `eval_data/breaking_examples_20260720.md`.
- **Routing swept all 6 Drive volumes** (2,391 pages): 2,192 deterministic ·
  101 fallback · 16 re-transcribe · 57 index · 25 admin.
  `eval_data/drive_routing_sweep_20260722.md`.
- **Free production build**: **5,235 production-ready sacramental records** with
  faithful text + provenance; 108 withheld for fallback; 3952 on the admin path.
  `eval_data/production_build_20260722.md`.
- **Partial-rate bug found and fixed (2026-07-24, Daniel's catch).** `partial`
  was inferred page-locally from closing-formula/signature regexes overfit to the
  St. Augustine volumes (the signature pattern hardcoded `O'Reilly|Hassett`), so
  it measured *lexicon coverage*, not truncation. Now decided positionally after
  stitching. **Measured on the real volumes: 10.6% (556) → 0.1% (7) of 5,235**,
  with record boundaries unchanged. The 7 survivors are 3 end-of-volume, 3
  title/cover pages, 1 untranscribable image. **Delivered corpus 4,679 → 5,228.**
- **Controlled vocabularies wired in** — `vocab.json` +
  `training_data_documentation.txt` vendored; the prompt's field spec is
  generated from `vocab.json` so it cannot drift. Baseline conformance of the
  pre-fix 701054 output: relationship_type 98.9%, occupation 99.4%, rank 100%
  (already fine) vs **age 26.4%** and **ethnicity 0%** (`crioulo` is a phenotype)
  — those two are the real available wins, not relationships. `witnesses` added.
  Measure with `vocab.is_known()`, NOT `canonicalize()` (which returns None by
  design for source-language fields and yields false 0%s).
- **Full pipeline validated end-to-end on real model output** (78 sample
  entries → 215 identities → 606-edge graph; priests O'Reilly ×39 / Hassett ×34
  correctly unified). `eval_data/pipeline_end_to_end.md`. `nodes.csv`/`edges.csv`
  are now exported beside the GraphML (701054: 330 nodes / 370 edges).
- **Engineering**: 155 offline tests (<1s, no network), reproducible builds,
  spend-safety rails, provenance throughout.

### 🟢 Live extraction — final Batch job in progress under a $20 cap (2026-07-23)
Approved capped Luna run via `run_luna_production.py` (hard cap, reserve-before-
send, validates stop/JSON/exact-IDs, `--confirm` required, key from env only).
Ledger `production/luna_live/spend_ledger.json`:

| Volume | State | Records |
|---|---|---:|
| **701054** (Portuguese) | ✅ **complete end-to-end** — extracted + QA + identity + graph | 212 |
| 176899 | 🟡 500 records materialized + QA/identity/graph; 587 records included in final provider job | 500 / 1,087 locally ready |
| 201991 · 29597 · 375062 | 🟡 included in final provider job | 0 / 3,936 locally materialized |

The first 176899 job returned 499 complete records and omitted one requested
entry. Exact-ID validation caught the omission; the one record was retried and
validated independently, yielding the 500-record partial demonstration. The
original incomplete job remains as an audit artifact and was billed.

**Final provider submission:** `batch_6a61ad7d08f88190968f330fb7d529b7` contains
the remaining **455** non-overlapping 10-record requests (4,523 records) in one
OpenAI Batch job. It is capped by an $18.20 reservation and may take up to 24
hours. A read-only monitor may download and validate it, but cannot submit any
additional paid work.

Spend at submission: **$1.6030095 confirmed + $18.20 reserved = $19.8030095 of
the $20 cap.** This leaves $0.1969905 for any individually approved repairs.
After exact-ID/JSON/stop/usage validation, the remaining outputs will be
materialized by volume and passed through the free QA, identity, and graph
pipeline. 701054 is already complete and materialized with faithful + normalized
text + people/events + provenance (`production/luna_live/701054.materialized.json`);
its QA flagged 11 possible duplicates + 5 chronology issues **for review, not
auto-edited**. Supervisor package: `production/luna_live/SUPERVISOR_DEMO_TOMORROW.md`.

### ✅ Resolved by Daniel's 2026-07-24 review
- **Schema approved** ("schema looks fine; I approve of only recording
  content-ful fields on a per-individual basis"), with the field docs and
  controlled vocabularies now vendored and wired into the prompt.
- **Trailing-partial convention — moot.** The 556 was the lexicon bug, not a
  convention question. Dropping page-truncated records now removes 7.
- **Cross-language question — withdrawn.** It was mis-framed on our side; none
  of these volumes mix Spanish and Portuguese, and the manual examples are
  consistent. Segmentation is validated cross-language (47/47); entity F1
  remains measured on Spanish, which Daniel did not flag as a concern.
- **Remaining volumes green-lit** by Daniel; extracting under the $20 cap.

### ❗ Open — needs a decision or human step
- **Re-extraction with the vocabulary-aware prompt is UNMEASURED.** The prompt
  fix is in `main` and the batches are re-staged
  (`production/batches_v2/`, +$0.02 vs the old prompt), but whether age climbs
  off 26.4% and ethnicity off 0% is a prediction until a run happens.
- **Poll/settle the final 455-request batch** (keyed step). Ledger still shows it
  `submitted` with the $18.20 reservation held, though Daniel observed $12.28
  billed at OpenAI — so it likely ran and needs validating + settling.
- **Cost reconciliation**: Daniel's $0.017/image used the 712 demo records as the
  denominator; across all 2,366 pages the full run is ~$0.005/image. Confirm once
  the batch settles.
- **Weak extraction dimensions → human review**: relationships (~0.83) and fine
  attributes route to the review queue (built, not yet run).
- **108 fallback records + 16 re-transcribe pages**: separate capped run /
  upstream re-transcription.
- **3952 administrative material**: QA/pilot only, not production-approved.

---

## 4. Model choice & cost (measured, not projected)

Entity-level F1 vs the GPT-4o reference set, pooled across 3 San Agustín volumes
(`score_entity_f1.py`; note: agreement with a model-generated reference, not
independent human truth):

| Model | People | Events | Relationships | Coverage |
|---|---:|---:|---:|---:|
| **GPT-5.6 Luna** | 0.973 | 0.971 | **0.829** | 86/88 |
| GPT-5.4 mini | 0.923 | 0.979 | 0.738 | 87/88 |
| Claude Haiku 4.5 | 0.947 | **1.000** | 0.757 | 88/88 |

**Luna** leads people and the hard relationships dimension and is the most
stable across volumes → selected. Cost for these 6 volumes: **~$15 Batch API**;
comfortably under the $0.01/image target. Full detail + caveats:
`eval_data/entity_f1_bakeoff.md`, `eval_data/llm_model_research.md`.

**Reasoning level — PINNED to `low` (2026-07-22).** `run_corpus_prompts.py`
now takes `--reasoning {minimal,low,medium,high}` (default **low**) and bakes
`reasoning_effort` into both the staged batches and the expanded Batch-API send
body (OpenAI only; omitted for Anthropic). `low` is chosen because extraction is
bounded rule-following (apply the normalization rules, fill the schema per a
fixed formula), not open-ended reasoning — a little headroom above
minimal/none for the inference-y relationship edges, without medium/high's
reasoning-token cost. **Caveat, stated honestly:** the F1 numbers above were
measured at the API *default* (unset). Before the full run, **confirm F1 at
`low`** on the staged validation sample (`production/validation_low/`, one
volume, ~$0.06 Batch API) with `score_entity_f1.py`; adjust the level if it
regresses. Reasoning tokens bill as output, so the ~$15 estimate (which assumes
~900 output tok/entry) is a floor at `low`.

---

## 5. Commands (Windows PowerShell; `python` is not on PATH)

```powershell
$py = 'C:\Users\mahajar\AppData\Local\Programs\Python\Python312\python.exe'

& $py -m pytest tests -q                                   # 122 tests, offline, <1s
& $py run_route_volume.py VOL.json --source-kind auto --out manifest.json   # $0
& $py run_production.py                                     # $0, all 6 volumes -> production/
# $0: stage priced Luna batches with reasoning pinned (already run -> production/batches/)
& $py run_corpus_prompts.py --corpus production/corpus --outdir production/batches `
      --model gpt-5.6-luna --reasoning low
# $0: expand a volume to the verbatim OpenAI Batch-API upload file (reasoning_effort baked in)
& $py run_corpus_prompts.py --expand production/batches/701054.batches.jsonl
# --- paid, only after: confirm F1 at `low` on the sample, then approve ---
#   OpenAI Batch API: upload production/batches/<vol>.batchapi.jsonl, poll, download.
& $py run_pipeline.py EXTRACTED.json --tag VOL --outdir out_vol   # $0, QA+identity+graph
& $py score_entity_f1.py                                   # $0, quality table from saved runs
```

Paid-run safety rules (enforced): dry-run first showing worst-case new spend +
cumulative ledger; explicit approval per changed plan; hard caps; keys read from
env only, never printed/committed; validate JSON/ids/usage after each call; keep
reservations on network/5xx, release only definitive unbilled 4xx.

---

## 6. The one thing blocking "complete"

The pipeline is **built, tested, and validated**; the free deterministic output
exists; the paid step is staged and priced. What remains is not more building —
it is **(a) sending this status to Daniel for sign-off**, which also settles the
open convention/cross-language questions, and **(b) running the ~$15 Luna
extraction** once approved, then `run_pipeline.py` on the result. After that, the
extracted, resolved, graphed dataset is the finished product.
