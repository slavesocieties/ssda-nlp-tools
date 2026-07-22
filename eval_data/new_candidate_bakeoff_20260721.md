# Eight-record candidate bake-off — 2026-07-21

Paid interactive tests used the same first eight held-out records from pages
0035–0044 and the same 15 examples. Each model had a cumulative hard cap of
$1. API keys were process-scoped and are absent from saved artifacts.

Gemini 3.6 Flash used minimal thinking. Claude Sonnet 5 used medium effort with
adaptive thinking disabled, correcting the output-allocation failure observed
in the earlier cached Batch experiment.

| Model | Parsed | Input | Output | Actual test cost | Cost/record |
|---|---:|---:|---:|---:|---:|
| Gemini 3.6 Flash | 8/8 | 14,877 | 4,549 | $0.056433 | $0.00705 |
| GPT-5.6 Terra | 8/8 | 14,340 | 4,064 | $0.096810 | $0.01210 |
| Claude Sonnet 5 | 8/8 | 22,346 | 6,625 | $0.110942 | $0.01387 |

## Same-eight-record agreement with the GPT-4o-generated reference

These are reference-agreement measurements, not independently verified truth.
Entity scores are F1. Normalized text uses `SequenceMatcher` with
`autojunk=False`.

| Model | People F1 | Events F1 | Relationships F1 | Normalized mean | Normalized minimum |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 0.9885 | 0.8889 | **0.8571** | 0.9825 | 0.9737 |
| GPT-5.6 Terra | **0.9885** | **1.0000** | 0.8529 | 0.9740 | 0.9518 |
| GPT-5.4 mini | 0.9655 | **1.0000** | 0.7429 | 0.9771 | 0.9686 |
| Claude Sonnet 5 | 0.9425 | **1.0000** | 0.7222 | **0.9838** | **0.9758** |
| Gemini 3.6 Flash | 0.9425 | **1.0000** | 0.7222 | 0.9706 | 0.9474 |
| Claude Haiku 4.5 | 0.9425 | **1.0000** | 0.6857 | 0.9776 | 0.9707 |

## Interpretation

Terra is the only new candidate that nearly matches Luna on the relationship
dimension while correcting Luna's one event disagreement in this small slice.
Sonnet has the closest normalized prose but does not improve people or
relationship extraction. Gemini 3.6 Flash is the cheapest new candidate and
now returns complete JSON, but its entity scores do not beat mini or Luna here.

At vendor Batch rates, a rough first-order production estimate is half the
interactive cost per record: Gemini $0.00353, Terra $0.00605, and Sonnet
$0.00693. This is not a corpus measurement; provider tokenization, cache hits,
and output length can change it. In particular, the prior Sonnet async Batch
test produced no prompt-cache reads, so its production estimate must not assume
reliable Batch cache reuse.

The eight-record sample is useful screening evidence, not enough to replace the
32-record/full-volume result. Extend Terra first if purchasing more evidence;
it is the only new model whose entity-quality signal is competitive with Luna.

## Terra extension to all 32 reference entries

The remaining 24 entries were run after explicit approval under the same
cumulative $1 Terra cap. All three additional requests completed with 8/8
parsed records:

| Added batch | Input tokens | Output tokens | Cost | Parsed |
|---|---:|---:|---:|---:|
| 2 | 14,330 | 4,351 | $0.10109 | 8/8 |
| 3 | 14,498 | 4,368 | $0.10177 | 8/8 |
| 4 | 14,522 | 4,605 | $0.10538 | 8/8 |

The extension cost $0.308235. Terra's complete 32-entry measured cost was
$0.405045, with no unresolved reservation.

| Model | Coverage | People F1 | Events F1 | Relationships F1 | Normalized mean | Normalized minimum |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Terra | 32/32 | **0.9917** | **1.0000** | **0.8435** | 0.9793 | 0.9518 |
| GPT-5.6 Luna | 32/32 | 0.9890 | 0.9620 | 0.8360 | **0.9842** | **0.9737** |

Terra slightly leads the GPT-4o-generated reference on the three structured
dimensions; Luna leads normalized-text agreement and costs materially less.
The structured differences are small, particularly relationships, and the
reference is not independently hand-verified. The result supports Terra as a
quality challenger but does not establish enough accuracy gain to displace
Luna as the current cost/quality choice.
