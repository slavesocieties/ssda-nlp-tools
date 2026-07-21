# Entity-level extraction F1 — live bake-off outputs vs gold

Scored with `run_eval.py` (name-aligned P/R/F1, no network) against the
hand-labeled gold `data` fields. This is the structured-data quality dimension —
people/events/relationships/attributes — **not** the normalized-text similarity
reported elsewhere. Regenerate the cross-volume table with `score_entity_f1.py`.

**Scope caveat:** all three reference volumes are San Agustín Spanish (~1794).
Cross-VOLUME validated; cross-LANGUAGE (Portuguese/Colombian) still open.
Attribute rows with small gold-n (ethnicity n=3, age n=7–10) are noisy.

## Cross-volume core F1 (2026-07-20, 3 volumes × 3 models)

Entries-weighted means across volumes 0013–0023, 0024–0034, 0035–0044:

| Dimension | gpt-5.6-luna | claude-haiku-4.5 | gpt-5.4-mini |
|---|---|---|---|
| people | **0.986** | 0.943 | 0.930 |
| events | 0.986 | **1.000** | 0.978 |
| relationships | **0.843** | 0.763 | 0.761 |

**Luna is the most stable.** Per-volume relationship F1: Luna 0.826 / 0.878 /
0.828 — tight. mini and Haiku **collapse to 0.60 / 0.67 on volume 0024_0034**
(verified NOT a parse artifact: 0 parse errors there; mini merely missed 1/29).
Relationships remain the weakest core dimension for every model — the typed
directed edges (enslaver/slave/godparent/parent) concentrate the errors and are
the review queue's main job. This cross-volume result triple-confirms the model
pick (cost + text-similarity + entity-F1 all favor Luna).

### Single-volume detail (0035_0044, all attributes below)

| Dimension | gpt-5.4-mini (n=32) | gpt-5.6-luna (n=24) | claude-haiku-4.5 (n=24) |
|---|---|---|---|
| people | 0.956 | **0.989** | 0.960 |
| events | 0.991 | 0.974 | **1.000** |
| relationships | 0.797 | **0.826** | 0.757 |
| event-date accuracy | 0.889 | **1.000** | **1.000** |

## Per-attribute accuracy on matched people (| hallucination rate)

| Attribute | gpt-5.4-mini | gpt-5.6-luna | claude-haiku-4.5 | verdict |
|---|---|---|---|---|
| titles | 0.983 \| .03 | 0.939 \| .00 | 0.980 \| .02 | strong |
| occupation | 0.969 \| .11 | 0.923 \| .04 | 0.923 \| .04 | strong |
| free | 0.894 \| **.20** | 0.956 \| **.26** | 1.000 \| **.27** | accurate when present but over-asserts on null |
| phenotype | 0.696 | 0.732 | 0.774 | mediocre |
| origin | 0.821 | 0.500 | 1.000 | high variance |
| legitimate | 0.238 | 0.933 | 0.267 | unstable |
| rank | 0.400 | 0.500 | 0.500 | weak (small n) |
| age | 0.200 \| **.68** | 0.286 \| **.67** | 0.143 \| **.65** | **unreliable — invents ages** |
| ethnicity | 0.000 | 0.000 | 0.000 | **fails (n=3, noisy)** |

## Takeaways

1. **What's production-ready now:** who is in each record (people), what
   sacrament happened (events), event dates, and the high-frequency attributes
   (titles, occupation). These clear ~0.95+.
2. **What needs a human-review loop or prompt work:** relationships (~0.80),
   and the fine demographic attributes — especially **age** (heavy
   hallucination), **free** (over-asserted on nulls), ethnicity, legitimate,
   rank. `run_review.py` already exists for exactly this queue.
3. **Model choice on entity quality:** gpt-5.6-luna leads people + relationships
   + dates; claude-haiku ties on people and is perfect on events but worst on
   relationships; gpt-5.4-mini is the solid middle on the largest sample.
   Consistent with the cost/text-similarity ranking.
