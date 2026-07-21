# Extraction-prompt improvements — proposal (needs A/B validation before adoption)

The cross-volume entity F1 and the end-to-end QA run surfaced three concrete,
prompt-addressable defect classes. Below are targeted additions to
`BATCH_SYSTEM_PROMPT` (`ssda_nlp_tools/batch_extract.py`), each tied to measured
evidence. **Not applied to the live prompt** — a production extraction prompt
should not change without an A/B test, and A/B testing needs paid calls (author
can't run them). The validation recipe is at the bottom; it costs ~$0.30/model.

## Defect 1 — dangling relationships (the #1 QA issue)

**Evidence:** 5 of 6 QA issues across 78 entries were `dangling_relationship` — a
person's `related_person` points at an id the model never emitted as a person
(e.g. P03 → missing P05). Purely an internal-consistency slip.

**Proposed rule (add under "Extraction rules"):**

> - RELATIONSHIP CLOSURE: every `related_person` id you reference MUST also
>   appear as a person object in the SAME entry's `people` list. If you record a
>   relationship to someone (parent, godparent, enslaver, spouse), you must
>   include that someone as a person — even if the text gives only a name. Never
>   point a relationship at an id you did not emit.

Low risk: it can only make output more self-consistent. This is also cheaply
enforceable in post-processing, so it doubles as a `run_qa.py` auto-repair
candidate (add the missing person stub, or drop the edge, flagged for review).

## Defect 2 — invented ages

**Evidence:** age accuracy ~0.14–0.29 with ~0.65 hallucination rate across all
three models — the worst attribute. Models infer an age band when the text
states none.

**Proposed rule (tighten the existing null-over-guess line):**

> - AGE: emit `age` ONLY when the entry explicitly states it (a number, "de un
>   mes", "parvulo", "adulto", etc.). Do NOT infer age from context (a baptism
>   does not imply "infant"; a marriage does not imply "adult"). If unstated,
>   `age` is null.

## Defect 3 — over-asserted enslaved/free status

**Evidence:** `free` accuracy is high when present but hallucination rate
~0.20–0.27 — the model asserts a status on people the gold left null.

**Proposed rule:**

> - FREE/ENSLAVED: set `free` ONLY on an explicit textual cue (esclavo/a,
>   libre, "esclava de", manumitido). Do NOT infer it from phenotype, origin,
>   surname, or an enslaver mention elsewhere in the entry. If unstated, null.

## Why not just ship it

Each rule is individually plausible but the interactions are unknown: tightening
age/free could suppress *correct* implicit extractions the gold actually rewards
(some gold values may be legitimately inferred). Only a gold-scored A/B run tells
us net F1 moved the right way. Relationship-closure is the safest of the three.

## Validation recipe (~$0.30/model, offline scoring)

1. Copy `BATCH_SYSTEM_PROMPT` to a variant with the three rules added.
2. Re-run `run_model_bakeoff.py` on the 3 reference volumes with the variant
   (a `--system-prompt-file` hook would need adding, ~10 lines).
3. Score both with `score_entity_f1.py`; adopt only rules whose dimension F1
   rises (relationships, age-accuracy, free-hallucination) without lowering
   people/events. Keep relationship-closure regardless if it doesn't regress.
