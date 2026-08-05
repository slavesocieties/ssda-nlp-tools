# SSDA NLP pipeline — project status & how it works
*Single source of truth. Last updated 2026-07-30. Supersedes the 2026-07-15 and
2026-07-20 supervisor drafts.*

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
- **Identity resolution** (`disambiguate.py`) — see 2.6; `run_pipeline.py` still
  invokes it, but it is now a stage in its own right.
- **Cross-chunk linking** (`link.py`) — unifies people across volumes.
- **Social graph** (`network.py`) — person registry + GraphML.
- **Review UI** (`review_html.py`) — a page to decide borderline merges.

### 2.6 Merging — `run_merge.py` (`disambiguate.py`), separate since 2026-07-29
Daniel: *"handle merging completely separately from extraction."* Extraction is
the only paid step and it is settled; merging is free and its rules are still
moving, so fusing them made every merge experiment look like it needed a
re-extraction. `run_merge.py` reads delivered extraction output and never writes
to it; re-running with different thresholds costs minutes and $0.

**The governing rule is his**: *"No people should be merged strictly based on
name correspondence; it should depend on a combination of date overlap,
same-named relation, same/similar qualities."* So nothing merges on a name.
Corroborating signals are **counted, not scored** — a score threshold let one
weak signal through, and shared register (which everyone in a volume has) was
clearing bars by itself. Signals: date overlap, same-named relation, matching
qualities, discriminative relation. Two are required for an exact surname, three
for a near variant, four for a distant one.

One sanctioned shortcut, also his: clergy in consecutive records merge on the
name, narrowly (clergy both sides, name similarity ≥ 0.92, within 12 pages).

Corpus effect vs. the pre-2026-07-29 behaviour: people 18,228 → **22,702**,
largest connected component 8,960 → **1,526**, cross-register edges 228 → **200**.
The last figure is the point — dissolving the giant cluster cost 28 of 228 real
links, so it was almost entirely an over-merging artefact.

### 2.7 Training the merge model — `build_training_sample.py`, `analyze_labels.py`
Rules have gone as far as they can. Measured: after all of the above, *María del
Rosario* still holds 31 mentions and *María de la Concepción* 45, because two
different women of that name, both parda, both free, baptised in the same decade,
genuinely do have two matching signals. No threshold separates them.

`build_training_sample.py` draws a stratified pair sample (2,000 pairs covering
**185/185** case types; a literal 10% would be 89,868 and ~250 review-hours), and
`likelihood_review_html.py` renders Daniel's 0/25/50/75/100 scale. When
`labels.json` returns, `analyze_labels.py` reports per-rule where each is **too
strict** and **too loose** — the former being the direction no internal
measurement can see, since a wrongly-refused merge fails silently.

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

### ✅ Validated V3 corpus delivery (2026-07-28)
The five approved corpus volumes (176899, 201991, 29597, 375062, 701054) are
complete end-to-end. `production/luna_v3/CORPUS_SUMMARY.json` records **5,228
delivered records**, **0 missing source records**, and **0 invalid batches**.
Seven page-truncated source records remain in the auditable deterministic corpus
and are excluded from delivery under the approved convention. The delivery keeps
the deterministic ID, image provenance, faithful transcription, normalized text,
and structured people/events data together.

Provider outputs passed normal-stop, exact-ID, JSON, project-schema, and usage
validation. Raw output is audit-only; `*.accepted.jsonl` is the only input to
assembly, so a malformed batch cannot reach the delivered corpus. The final V3
cost is **$24.5323885 confirmed** of the $35 cap; no V3 reservation remains.
QA, identity, graph, and review artifacts live in `production/luna_v3/`;
`SUPERVISOR_RESULTS.md` is the concise handoff.

### 🟢 Approved additional extraction — submitted, awaiting validation (2026-07-30)
Only Daniel-approved volumes 701157 and 701179 were staged in
`production/approved_extract/`; the three exploratory volumes are excluded.
Both staged files use `gpt-5.6-luna` with `reasoning_effort=low`, and their entry
IDs were checked against the delivered corpus with **zero collisions**.

| Volume | Source entries | Requests | Batch job | State |
|---|---:|---:|---|---|
| 701157 | 873 | 88 | `batch_6a6be918838c81908f35e07fc553d51f` | submitted; validate before assembly |
| 701179 | 697 | 70 | `batch_6a6be91b5d8481908a93b337619d89ff` | submitted; validate before assembly |

The two jobs hold a conservative **$6.32 reservation**. Together with the
validated V3 spend, this is **$30.8523885 of the $35 cap**, leaving
**$4.1476115**. The staged cost projection is $4.52; the ledger will be settled
only from provider-reported usage after all requested IDs and schemas validate.

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
- **Additional volumes approved** by Daniel. The 701157 and 701179 Batch jobs
  are submitted under the shared $35 cap and awaiting provider validation; see
  the current-state table above.

### ❗ Open — needs a decision or human step
- **Validate and assemble 701157 + 701179.** Both are already submitted. Do not
  assemble them until normal-stop, exact-ID, JSON, project-schema, and usage
  checks settle their shared-ledger reservations.
- **Upstream transcription decision: retain Gemini-3.1-Pro.** A valid,
  same-local-image Luna probe on `701157-0056.jpg` was completed and inspected.
  Evidence under `production/bakeoff/luna_701157_0056_probe/`.

  Luna did not misread the page, it **confabulated** one. All three of its
  entries used a formula absent from the folio (`em casas de morada de <name>`),
  with dates in an invented sequence (23/26/28 Nov **1842**) and three
  householders who do not appear. Reading the image directly, the folio holds
  three marriages dated **29 Nov 1841, 18 Nov 1841, 12 Jun 1841**. Downstream
  screen 5–1 to Gemini; text similarity 0.4381.

  Two honest qualifications. Gemini is better, not clean: right on entries 1 and
  3, but it reads *Dezembro* where the manuscript plainly says *Novembro* on
  entry 2, and its 4-vs-3 entry "win" is partly spurious — the extra entry is a
  right-margin annotation mis-segmented as a record. And this is **one page**.

  **The finding that generalises is about our own screen, not about Luna.**
  Every free metric in `transcription_bakeoff.py` measures *well-formedness*,
  not fidelity, and fluent fabrication is well formed — Luna actually **won**
  the vocabulary metric with invented text. `formula_rate` caught it only by
  luck, because Luna invented a formula the regex does not list. Had it
  fabricated with the correct opening formula, the screen would have read
  near-even on a page where one model made everything up. The screen ranks where
  to look; only the image certifies accuracy. The same bound applies to
  `repair_burden()`: confabulated text needs no repair, so it scores as
  *cleaner*.

  A four-page Luna script (`run_probe_set.py`, 1701–1907, both languages) exists
  as a guarded confirmation experiment, not a recommended production change.
  Every confirmed upstream call now needs an explicit USD reservation,
  persistent ledger, and hard cap.
- **Vocabulary-aware extraction is measured for age.** On held-out Portuguese
  701054, age-category conformance rose from 26.6% (46/173) to 100.0%
  (173/173). Ethnicity remains an open historical descriptor field; it is
  preserved verbatim and routed to its term-level review queue, not judged by a
  closed-vocabulary percentage.
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

## 6. What is left for this to be complete (updated 2026-08-05)

Verified against artifacts on the date above, not carried forward from the
previous revision. Every figure here is reproducible: `audit_corpus.py`,
`validate_graph.py`, `verify_claims.py`, `verify_label_scores.py`.

**Where the corpus actually is.** 6,794 records across 7 volumes (the previous
revision said 5,228 / 5 -- 701157 and 701179 were assembled on 2026-08-03 from
extraction already paid for). 8,595 events: baptism 2,815, burial 2,172, birth
2,026, marriage 1,582. Graph: 32,628 people, 53,800 edges.

### What "complete" still requires

**A. Blocked on Daniel. Nothing here can be moved by building.**

1. **The 300 synthetic pairs** (`synthetic_pairs.html`, sent 2026-08-05). This is
   still the critical path. Merge agreement against his 25 real labels is
   21/24 (88%) on the pairs he was certain about -- a smoke test with a real
   signal, not a benchmark.
2. **The double-surname ruling.** 534 distinct name pairs over 1,982 mentions
   (5.0% of the corpus), of which we merge ZERO. 75% is clergy and looks
   unambiguous; the lay quarter is a different question because the trailing
   token is often an ethnonym (Lucumi, Congo), not a surname.
3. **The corroboration bar.** An exact surname match still needs 2 signals; 144
   of 231 exact-surname pairs in the sample are refused for want of one. This is
   his own ruling doing most of the work of the merge bar.
4. **Burial gold.** 12 candidates await his correction. Burials are 2,172 events,
   25% of the corpus, and **burial accuracy has never been measured at all.**
5. **Portuguese vocabulary.** 966 Portuguese titles (Reverendo 441, Coadjutor
   325) and Brazilian ethnonyms; ethnicity conformance fell to 61%/45% on the
   first substantially Portuguese volumes. Needs a ruling like the 71 ethnicity
   terms.
6. **180 free repairs** (282 issues over 180 distinct records: 172 duplicate
   entries, 110 null relationships). They change delivered counts, so they are
   staged rather than applied.

**B. Blocked on a human step or spend.**

7. **API key rotation** -- outstanding across several sessions.
8. **244 records need re-extraction** -- 3.6% of the corpus. (That is 363
   ISSUES over 244 DISTINCT records; one record commonly carries several
   dangling relationships. Earlier revisions quoted the issue count as if it
   were records.) Source text is fine; the extractor misread it. PAID.
   Concentrated: 375062 at 7.8% and 29597 at 7.3% supply 145 of the 244 from
   28% of the corpus, six times 201991's 1.3%. See RUNBOOK §18, §20, §22.
   **29597 is explained** (densest entries in the corpus; re-extraction should
   help). **375062 is diagnosed and is NOT one purchase**: ~18 of its 88 have a
   broken normalisation as INPUT -- empty, or model commentary instead of text --
   so re-extracting them re-reads the same broken input and cannot fix anything.
   Those need re-normalisation or re-transcription. Another ~23 returned
   {"people": [], "events": []} on clean, well-formed baptism text, where a
   retry is reasonable.
9. **701054 completion** -- NOT a re-extraction. 375 records were never
   extracted once, because our segmenter dropped 54 of its 105 pages as "index
   pages" (two-column registers transcribed as markdown tables). 61 requests
   staged at `production/batches_v6/`, ~$2.10. One-time; the bug is fixed.
   Needs a fresh outdir and distinct run-id: 210 of 596 entry ids collide.

**C. Unblocked -- can be built now.**

10. ~~Residual graph defects~~ **DONE for the fixable part** (2026-08-05).
    `missing_inverse` 6 -> 2 by materialising SELF-INVERSE types (sibling,
    spouse, witness), where the reverse is definitional rather than inferred.
    Rebuilt graph at `corpus_final_pipeline_v2/` (32,943 people, 53,844 edges);
    self-loops and dangling endpoints remain 0.

    What is left is not a graph bug and cannot be fixed in the graph:
    - **2 missing inverses + 1 role contradiction are ONE malformed entry**,
      `701179-0148-01`, where Maria and Antonio da Costa are each recorded as
      the other's parent AND godparent. Needs RE-EXTRACTION (item 8).
    - **4 contradictory role pairs and 15 ancestry cycles are merge artifacts**,
      not extraction errors -- `same_entry_role_contradictions` finds exactly 1
      in 6,794 records. The cycle guard is depth-bounded to 4 and reroutes some
      loops rather than removing them; fixing that properly needs labels.

    NOT promoted over `corpus_final_pipeline/`: the delivered graph is a
    hand-off artifact and replacing it is a call for a human to make.
11. ~~The review queue for weak dimensions~~ **BUILT AND RUN** (2026-08-05).
    It had never been built -- the four review modules are all identity/pair
    review and the only queue produced was for ethnicity TERMS.
    `run_relationship_review.py` -> 331 rows, all shown: 216 dangling
    relationships, 110 null, 4 dangling principals, 1 role contradiction.
12. ~~Statistical validation of the training sample~~ **DONE** (2026-08-05).
    `validate_training_sample.py`: reservoir uniform (chi-square 477.6 vs a
    4-sigma bound of 625.4, by simulation), water-fill spread 1, and
    sum(weights) = 7,305,667 against a population of 7,305,667 -- 0.000% error,
    which also proves all 444 strata drew at least once. 37 drew exactly once.

### The honest ceiling

**Extraction is the only stage with no human ground truth**, and it sits
upstream of everything downstream. `slavesocieties/openai` has ~86 entries with
structured data, but ~65 are model output in `testing/`; only ~21 are usable.
Measuring our extraction against another model's extraction measures agreement,
not accuracy. Burials are the sharpest case: a quarter of all events, never
measured once.

Everything below extraction is measured:

    transcription   substitution 6.31%, median page similarity 0.891 (3 vols)
    segmentation    98.2% of the human entry count (3 vols)
                    ^ those 3 gold volumes (15834, 1795, 419324) are DISJOINT
                      from the 7 delivered. Evidence about the pipeline, not
                      about the shipped data. See RUNBOOK section 21.
    normalization   age 100%, ethnicity 100% on Spanish volumes
    merge           21/24 (88%) against Daniel's certain labels
    graph           invariants above

So "complete" is not a build target. It is: extraction measured against human
truth, burials included, and the merge bar set by labels instead of by argument.
