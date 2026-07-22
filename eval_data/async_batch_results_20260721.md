# Async Batch API results — Gemini 3.5 Flash and Claude Sonnet 5

Polled 2026-07-21 from three already-submitted jobs. Polling was read-only; no
new inference work was submitted. Full provider responses remain gitignored.
This report retains only usage and validity summaries.

## Outcome

| Job | Requested | Valid complete JSON | Stop/result | Measured or bounded cost |
|---|---:|---:|---|---:|
| Gemini 3.5 Flash Batch | 8 records | **0/8** | `MAX_TOKENS`; truncated JSON | **$0.05609** |
| Sonnet 5 Batch, uncached | 8 records | **0/8** | `max_tokens`; truncated JSON | **$0.07235** |
| Sonnet 5 Batch + 1h prompt cache | 8 records | **1/8** | one incomplete batch ended; one truncated | **≥$0.12549**, ≤$0.25575 reserved ceiling |

None produced a usable held-out sample, so there is no defensible quality F1
for these jobs. They are eliminated in their tested configurations.

## Sonnet 5 prompt-caching result

The synchronous warm-up reported a 19,573-token cache read, but that cache was
not reused when the asynchronous Batch requests executed:

| Batch usage | Value |
|---|---:|
| ordinary input tokens | 2,817 |
| 1-hour cache creation tokens | **41,134** (20,567 × 2) |
| cache read tokens | **0** |
| output tokens | 7,299 |
| thinking tokens (included in output) | 6,606 |

Anthropic documents that cache entries become available only after the first
response begins and that Batch work may take long enough for ephemeral cache
entries to expire. The observed result is consistent with that limitation: the
two async requests created the same cache independently instead of reading it.

At the introductory Sonnet 5 rates through 2026-08-31, Batch costs $1/M input
and $5/M output. A 1-hour cache write is 2× input and a cache read is 0.1×;
those multipliers stack with the Batch discount. The two Batch results therefore
cost $0.12158. The warm-up's known cache-read component adds $0.00391; its small
ordinary input/output usage was not saved, hence the reported **$0.12549 floor**
rather than false precision.

The main failure was output allocation: Sonnet 5 uses adaptive thinking by
default. It consumed 6,606 of 7,299 output tokens on thinking, leaving too little
room for eight structured records. Raising the cap would increase cost but would
not fix the unreliable cache-hit behavior.

## Gemini 3.5 Flash result

Provider usage was 14,877 input tokens, 4,172 candidate tokens, and 5,812
thinking tokens. Gemini Batch pricing is $0.75/M input and $4.50/M output,
including thinking, giving **$0.05609**. The response stopped at the token limit
with malformed/truncated JSON, so it supplies no quality result.

## Decision

Do not select Sonnet 5 on the assumption that Batch + prompt caching will make
this workload cheap. The live test produced no Batch cache hits and only 1/8
valid records. Do not select Gemini 3.5 Flash from this run either: it produced
0/8 valid records. Luna and Haiku remain the viable quality candidates, with
mini the high-coverage lower-cost comparison.

Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing),
[Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
[Anthropic Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing),
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).
