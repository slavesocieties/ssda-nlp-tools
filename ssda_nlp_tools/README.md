# ssda_nlp_tools — local NLP tooling (measurement + disambiguation + fixes)

Local, self-contained tooling built on top of the SSDA NLP stage (Zekai's
`transcription_json_to_training_llm_repair.py` → `normalize.py` → `extract.py`).
**Nothing here calls an LLM or touches the network** — it scores and transforms
the JSON those scripts already produce, so it runs anywhere with just Python
(stdlib only: `unicodedata`, `difflib`). This is a *working copy*; it does not
modify the upstream repos.

## Why these three pieces

Reading the pipeline surfaced two high-leverage gaps and a set of latent bugs:

1. **Nothing measured extraction accuracy.** The `Reduction_test/` folder was
   hand-eyeballed model bake-offs. → an **evaluation harness**.
2. **Cross-entry person resolution was manual** (`utility.py:disambiguate_people`
   blocks on `input("y/n")` per pair) and **not even wired into Zekai's flow**,
   so the same enslaver across 40 records stays 40 separate people. → **automated,
   confidence-scored disambiguation** with a review queue.
3. **`parse_date` / `complete_date` were broken** (returned strings; crashed on
   ranges and on string months) and **`fix_relationships` only inspected the first
   event**. → corrected, unit-tested **`fixes.py`**.

## Layout

```
ssda_nlp_tools/
  textmatch.py    name normalization, phonetic folding, similarity, alignment, P/R/F1
  evaluate.py     gold-set eval: people / attributes / events / relationships
  disambiguate.py cross-entry identity clustering (phonetic blocking + union-find
                  + review queue + domain guards + human constraints)
  resolve.py      apply disambiguation back onto a volume -> global_id on every mention
  network.py      build the person/relationship social graph; export GraphML + summary
  link.py         combine chunks/volumes and link people ACROSS them
  qa.py           per-volume data-quality report (duplicates, chronology, refs, drift)
  review_html.py  self-contained review page + decisions -> constraints
  cost.py         token/cost model + optimizer to hit a $/image target
  batch_extract.py batched, cache-ordered, single-pass extractor (the cost recipe)
  segment.py      Task 3: deterministic Archivault->entry segmentation ($0/image)
  segeval.py      segmentation scoring + margin-number structural validation
  fixes.py        corrected parse_date / complete_date / is_principal / fix_relationships
run_pipeline.py         ONE COMMAND: qa -> link -> resolve -> network -> review page
run_eval.py             CLI over evaluate
run_disambiguate.py     CLI over disambiguate
run_network.py          CLI: disambiguate -> resolve -> network in one shot
run_link.py             CLI: cross-chunk/volume linking + unified graph
run_qa.py               CLI: data-quality reports
run_review.py           CLI: make review.html / apply decisions.json
run_cost.py             CLI: cost model + $/image optimizer + lever waterfall
run_batch.py            CLI: batched-extraction token saving + emit ready messages
run_segment.py          CLI: Task 3 segmentation (+ --structural / --eval checks)
run_corpus_prompts.py   CLI: segmented corpus -> priced, ready-to-send extraction
                        batches (compact JSONL; --expand emits OpenAI Batch-API files)
tests/                  67 offline tests (incl. Daniel's gold pairs as fixtures)
eval_data/model_agreement_0013_0023.{md,json}   compiled model bake-off report
eval_data/cost_to_penny.md                      how to hit $0.01/image (with recipe)
eval_data/task3_segmentation.md                 Task 3 results: gold-perfect, 100% structural
```

The full chain: **extract.py output → QA → disambiguate → resolve → link → network**,
with a human review loop (review.html → decisions.json → constraints). Each step is
independently runnable and testable. `run_pipeline.py --outdir out` does it all.

## Domain guards in the identity scorer (what makes merges trustworthy)

1. **Once-in-a-lifetime sacrament guard** — two baptism/birth/burial *principals*
   from different entries are different people (you are baptized once). This
   alone fixed every "three infants named María Dolores became one person" case,
   and it surfaces double-recorded entries in the review queue.
2. **Discriminative relationship context** — your enslaver, spouse, and parents
   don't change per entry. Same-typed context pointing at *different* third
   parties ("slave of Sánchez" vs "slave of McQueen") outweighs any pile of
   attribute agreements; matching context corroborates.
3. **Estate-surname awareness** — everyone attached to an estate shares its
   surname ("Hanna Macqueen" vs "Rachael Macqueen" are different wives), so
   third-party names compare given-name-first, with short forms ("Rachael" =
   "Rachael Macqueen") handled by token containment.
4. **Bare-name cap** — "Juan" ~ "Juan" with no context corroboration cannot
   auto-merge (population-universal attributes like phenotype don't count);
   it goes to review. This also stops context-empty mentions from acting as
   transitive union-find bridges between genuinely different people.

All four verified against real volume 239746 output (the Juan/Smart/María
Dolores cases in tests and in `eval_data/model_agreement_0013_0023.md`).

## Cost: transcription + normalization ≤ $0.01/image

`cost.py` measures token counts from the repo's own files (extract.py system
prompt, instructions.json, training_data.json, a real volume) and models every
lever. The diagnosis: the ~11k-token few-shot prefix is billed **once per entry**
today, and normalization is a **second full LLM pass**. The recipe (all three
quality-preserving; full analysis in `eval_data/cost_to_penny.md`):

1. **Prompt caching** — static prefix first and byte-identical across calls.
2. **Batch 10–20 entries/call** — pay the prefix once per batch (**8× fewer
   input tokens** on the real chunk).
3. **Fold normalization into extraction** — one call returns `normalized` + `data`
   (eliminates a whole pass; the driver's `LLM_REPAIR_RETURNS_NORMALIZED` hook).

Do **not** cut few-shots to save money — the bake-off shows it drops people-F1
for ~$0.0001. On gpt-4o-mini this takes total cost **$0.0108 → $0.0020/image**
(5×; $8k → $1.5k over 750k images) with transcription+normalization at **$0.0007**.
`batch_extract.py` implements 1–3 and is verified offline (cache-ordering,
response round-trip, token saving). `run_batch.py --emit` writes a ready-to-send
`messages` array.

## The human-in-the-loop cycle

```bash
python run_review.py make VOLUME.json --html review.html --tag V1
# open review.html (no server): s = same, d = different, u = unsure, j/k = move
# click "Download decisions.json"
python run_review.py apply VOLUME.json decisions.json --graphml net.graphml
```

Decisions become **must/cannot constraints**: must-links merge outright, decided
pairs leave the queue, and a cannot-link that a transitive chain still violates
flags the cluster instead of being silently ignored. Progress survives browser
restarts (localStorage).

## Usage

```bash
# THE one command — QA + link + resolve + network + review page, all artifacts:
python run_pipeline.py Sample_output/Generated_0013_0023_4o_prompt_V2.json \
                       Sample_output/Generated_0024_0034_4o_prompt_V2.json \
                       Sample_output/Generated_0035_0044_4o_prompt_V2.json \
                       --tags p13-23 p24-34 p35-44 --tag V239746 --outdir out_v239746

# Or piecemeal:
# 1) Evaluate a run against the gold set (training_data.json is 15 labeled entries)
python run_eval.py training_data.json Sample_output/Generated_0013_0023_4o_prompt_V2.json --errors

# ...or compare two model runs on the same pages (agreement report, no gold needed)
python run_eval.py Sample_output/Generated_0013_0023_4o_prompt_V2.json \
                   Reduction_test/Generated_0013_0023_less_ex_5.4.json

# 2) Data-quality report (duplicate entries, chronology, dangling refs, drift)
python run_qa.py Sample_output/Generated_0024_0034_4o_prompt_V2.json

# 3) Disambiguate people across a volume's entries
python run_disambiguate.py Sample_output/Generated_0013_0023_4o_prompt_V2.json \
                           --tag SSDA0013 --out identities.json

# 4) Resolve global ids + build the social graph (GraphML)
python run_network.py Sample_output/Generated_0013_0023_4o_prompt_V2.json \
                      --tag SSDA0013 --graphml net.graphml --json net.json --resolved resolved.json

# 5) Link people ACROSS chunks/volumes into one registry + graph
python run_link.py Sample_output/Generated_00*.json --tag V239746 --registry registry.json

# 6) Human review loop
python run_review.py make Sample_output/Generated_0013_0023_4o_prompt_V2.json --html review.html

# 7) Cost: model the $/image and get the recipe to hit $0.01 (see eval_data/cost_to_penny.md)
python run_cost.py --target 0.01
python run_batch.py Sample_output/Generated_0013_0023_4o_prompt_V2.json --instructions instructions.json --batch 10

# 8) Tests
python -m pytest tests -q
```

## The network deliverable (network.py)

Once mentions are resolved to volume-wide identities, per-entry relationships
collapse into ONE directed graph: nodes = people (with merged attributes +
mention count), edges = typed relationships (parent/child, godparent/godchild,
enslaver/slave, spouse) carrying weight and entry provenance. Exports **GraphML**
(loads in Gephi, networkx, Cytoscape) plus a summary JSON with degree hubs,
connected components, and — the number that matters — **cross-entry people/edges**
(the links a single entry could never give you).

On the real 0013–0023 sample this surfaces recurring **godparents** as social
hubs (e.g. a godmother linked to many baptisms) — exactly the cross-record
structure the archive exists to reconstruct. It also exposes a **schema gap**:
the officiating priest is 21 mentions but 0 edges, because officiants are stored
as people yet never linked to their events — worth considering as an added edge
type upstream.

## What the eval measures

Person IDs (`P01…`) are arbitrary and differ between gold and prediction, so the
harness **aligns people by name first**, then maps every id-referencing field
(event principals, relationship endpoints) through that alignment into a shared
gold-name space before scoring. Reported per run:

- **people** P/R/F1 (name-aligned detection)
- **attributes** per-field accuracy *and* hallucination rate on matched people
  (occupation, phenotype, free, origin, ethnicity, age, legitimate, rank, titles)
- **events** P/R/F1 (type + principals + date) and date accuracy on matched events
- **relationships** P/R/F1 over directed typed edges

Self-validated: gold-vs-gold scores a perfect 1.0; controlled perturbations
(drop a person, flip an attribute, empty predictions) move the metrics in the
expected direction — see the tests.

## Disambiguation scoring

`pair_score = name_similarity` adjusted by **attribute compatibility** (hard
conflicts like free-vs-enslaved or differing phenotype push apart; agreement
pulls together) and **shared-relationship context** (two people enslaved by the
same named enslaver are more linkable). Mentions from the *same* entry are never
merged. Confident pairs (`≥ auto`) auto-merge via union-find; borderline pairs
(`[review, auto)`) go to a ranked review queue. Chained (transitive) merges whose
weakest internal link dips below `auto` are flagged for review.

## Known limitations / next upgrades

- Similarity now includes a Spanish-aware **phonetic fold** (pure Python) used for
  both blocking and a discounted similarity signal; a true double-metaphone via
  `jellyfish` + `rapidfuzz` is the natural drop-in, isolated behind
  `textmatch.phonetic_fold` / `name_similarity`.
- **Common-name over-merge**: very frequent names (e.g. "María Dolores") with no
  distinguishing attributes can auto-merge into one inflated node. That's the
  precision/recall knob — raise `--auto`, or lean on the `needs_review` flag and
  the review queue. A stronger fix is attribute- and relationship-aware blocking.
- Entry alignment in the eval falls back to text similarity when ids differ;
  segmentation that splits/merges entries differently is reported as unaligned
  rather than partially credited.
- Disambiguation + the network are **within-volume**; cross-volume linking (the
  same person across decades/books) is the larger research problem and builds on
  the same `person_id` join key.
- The officiant→event link is not modeled as an edge (see above) — a schema
  decision to make upstream if officiant networks matter.
- The gold set is 15 entries — enough to validate the harness and catch
  regressions, not to certify absolute accuracy. Growing it is the obvious next step.
```
