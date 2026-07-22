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
  faithful text + provenance (556 truthful partials); 108 withheld for fallback;
  3952 on the admin path. `eval_data/production_build_20260722.md`.
- **Full pipeline validated end-to-end on real model output** (78 sample
  entries → 215 identities → 606-edge graph; priests O'Reilly ×39 / Hassett ×34
  correctly unified). `eval_data/pipeline_end_to_end.md`.
- **Engineering**: 122 offline tests (<1s, no network), reproducible builds,
  spend-safety rails, provenance throughout.

### 🟡 Staged — one approval away, not run
- **Paid extraction of the 5,235 records**: 527 Luna calls, **~$15.07 via Batch
  API** (~$30.21 interactive+cache). Prepared and priced; **not sent**. Run =
  dry-run → your approval → `--confirm`.

### ❗ Open — needs a decision or human step
- **Supervisor sign-off**: Daniel has not yet reviewed the output/schema. Nothing
  is "accepted" until he confirms it meets his needs.
- **Cross-language — narrowed (2026-07-22, per Daniel).** Portuguese (65858,
  260950) and Colombian (420550, 544367) examples **do** exist and are part of
  the 47/47 — so **segmentation is validated cross-language.** They are
  *segmentation* gold (`id/text/images`), though; every file carrying entity
  `data` (people/events) is Spanish. So the only remaining gap is **entity-
  extraction F1 measured on Spanish only** — lower-risk, since the hard
  structural stage is validated cross-language and extraction uses a general
  multilingual model. A Portuguese/Colombian entity-gold example would close it
  fully but is optional.
- **Weak extraction dimensions → human review**: relationships (~0.83) and fine
  attributes (age, enslaved/free, ethnicity) are unreliable and route to the
  review queue (built, not yet run). `eval_data/prompt_improvements_proposal.md`
  has targeted, untested prompt fixes.
- **Trailing-partial convention**: whether references that omit a page-truncated
  final record should be scored as such — a convention question for Daniel.
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
