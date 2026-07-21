# Entity-level extraction F1 — live bake-off outputs vs reference

Scored offline with the name-aligned evaluator against the existing
`Generated_*_4o_prompt_V2.json` reference fields. The repository does not
document independent human verification of those fields, so these numbers are
**agreement with GPT-4o-generated references**, not accuracy against confirmed
human gold. This is the structured-data quality dimension—
people/events/relationships/attributes — **not** the normalized-text similarity
reported elsewhere. Regenerate the cross-volume table with `score_entity_f1.py`.

**Scope caveat:** all three reference volumes are San Agustín Spanish (~1794).
Cross-VOLUME validated; cross-LANGUAGE (Portuguese/Colombian) still open.
Attribute rows with small gold-n (ethnicity n=3, age n=7–10) are noisy.

## Cross-volume end-to-end core F1 (corrected 2026-07-21)

True micro-F1, pooling TP/FP/FN across volumes 0013–0023, 0024–0034,
0035–0044. Missing model entries count as false negatives.

| Dimension | gpt-5.6-luna | gpt-5.4-mini | claude-haiku-4.5 |
|---|---|---|---|
| people | **0.925** | 0.923 | 0.898 |
| events | 0.916 | **0.979** | 0.939 |
| relationships | **0.789** | 0.738 | 0.718 |
| entry coverage | 78/88 (88.6%) | **87/88 (98.9%)** | 80/88 (90.9%) |

Luna retains the strongest relationship result, while mini leads events and has
near-complete coverage. Relationships remain the weakest core dimension for
every model. Because coverage differs and the references are model-generated,
this result supports Luna as the relationship-quality candidate but does not by
itself settle production model choice.

### Single-volume detail (0035_0044, all attributes below)

| Dimension | gpt-5.4-mini (32/32) | gpt-5.6-luna (24/32) | claude-haiku-4.5 (24/32) |
|---|---|---|---|
| people | **0.956** | 0.853 | 0.828 |
| events | **0.991** | 0.826 | 0.851 |
| relationships | **0.797** | 0.714 | 0.656 |
| event-date accuracy | 0.889 | **1.000** | **1.000** |

Attribute and date rows below are conditional on matched people/events and do
not penalize missing entries; interpret them together with coverage.

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
3. **Model choice on entity quality:** Luna leads end-to-end relationship F1;
   mini leads event F1 and coverage; their people F1 is effectively tied on this
   small, single-region reference. Haiku trails both overall.
