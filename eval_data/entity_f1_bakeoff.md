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
| people | **0.973** | 0.923 | 0.947 |
| events | 0.971 | 0.979 | **1.000** |
| relationships | **0.829** | 0.738 | 0.757 |
| entry coverage | 86/88 (97.7%) | 87/88 (98.9%) | **88/88 (100%)** |

Luna leads people and relationships, while Haiku has perfect event F1 and full
entry coverage. Relationships remain the weakest core dimension for every
model. The references are model-generated and single-region, so this result
supports Luna as the relationship-quality candidate but does not replace human
gold or cross-language validation.

### Missing 0035 batch completed live (2026-07-21)

The eight-record 0035 b1 gap was closed with one explicitly approved capped
request per model. Both returned 8/8 parseable records:

| Model | Live cost | Cost/record | Cumulative ledger | Cap |
|---|---:|---:|---:|---:|
| gpt-5.6-luna | $0.04041 | $0.00505 | $0.57866 | $1.00 |
| claude-haiku-4.5 | $0.05210 | $0.00651 | $0.69670 | $1.00 |

This removes the unequal-exposure artifact from volume 0035. The remaining
missing outputs are genuine returned-output gaps from other volumes: Luna 2 on
0013 and mini 1 on 0024. Haiku returned all 88 entries.

### Single-volume detail (0035_0044, all attributes below)

| Dimension | gpt-5.4-mini (32/32) | gpt-5.6-luna (32/32) | claude-haiku-4.5 (32/32) |
|---|---|---|---|
| people | 0.956 | **0.989** | 0.970 |
| events | 0.991 | 0.962 | **1.000** |
| relationships | 0.797 | **0.836** | 0.776 |
| event-date accuracy | 0.889 | **1.000** | **1.000** |

Attribute and date rows below are conditional on matched people/events and do
not penalize missing entries; interpret them together with coverage.

## Per-attribute accuracy on matched people (| hallucination rate)

| Attribute | gpt-5.4-mini | gpt-5.6-luna | claude-haiku-4.5 | verdict |
|---|---|---|---|---|
| titles | 0.983 \| .03 | 0.934 \| .00 | 0.984 \| .06 | strong |
| occupation | 0.969 \| .11 | 0.939 \| .03 | 0.941 \| .03 | strong |
| free | 0.894 \| **.20** | 0.971 \| **.21** | 0.985 \| **.19** | accurate when present but over-asserts on null |
| phenotype | 0.696 | 0.771 | 0.763 | mediocre |
| origin | 0.821 | 0.667 | 1.000 | high variance |
| legitimate | 0.238 | 0.952 | 0.286 | unstable |
| rank | 0.400 | 0.500 | 0.500 | weak (small n) |
| age | 0.200 \| **.68** | 0.300 \| **.66** | 0.200 \| **.64** | **unreliable — invents ages** |
| ethnicity | 0.000 | 0.000 | 0.000 | **fails (n=3, noisy)** |

## Takeaways

1. **What's production-ready now:** who is in each record (people), what
   sacrament happened (events), event dates, and the high-frequency attributes
   (titles, occupation). These clear ~0.95+.
2. **What needs a human-review loop or prompt work:** relationships (~0.80),
   and the fine demographic attributes — especially **age** (heavy
   hallucination), **free** (over-asserted on nulls), ethnicity, legitimate,
   rank. `run_review.py` already exists for exactly this queue.
3. **Model choice on entity quality:** Luna leads people and relationship F1;
   Haiku leads events and coverage; mini remains the lower-cost comparison but
   trails Luna on the archive's hardest dimension. **Recommendation: Luna with
   the existing QA/review queue; Haiku is the fallback when complete output is
   valued above relationship accuracy.**
