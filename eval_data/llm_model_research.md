# Which LLM for extraction: cost vs. precision (verified 2026-07-16)

Requested by Ronak, after a prior report on this topic (from a different agent,
working in a parallel copy of this repo) turned out to be unverifiable — no
sources, no benchmark evidence for its precision claims, and one confirmed
factual error (see §0). This version cites everything and computes cost
projections through our own pipeline's real, measured token counts — not
hypothetical ones.

## 0. What was wrong with the earlier report

The earlier report asserted specific model names, tiers, and prices with no
visible search, no sources, and no benchmark data for its precision claims
("flagship-level architectures... industry leader for structured JSON",
asserted with zero citations). Checking it properly:

- **The pricing numbers were mostly right** — genuinely surprising, and I was
  too quick to first dismiss "GPT-5.6 Terra" as invented before I'd checked.
  Correction acknowledged and verified against primary sources below.
- **One factual error survives**: it describes Claude Fable 5 as "Mythos-class."
  Anthropic's own pricing page lists **Fable 5 and Mythos 5 as two separate
  models** that currently share pricing — not one model with that name.
- **It never once addressed precision with evidence.** Every capability claim
  was asserted, not sourced. That's the actual gap this report closes.

Trust-but-verify held here: the specific facts were checkable, I checked them,
most held up, one didn't. That's a different thing from "the report was fine" —
a report with zero citations that happens to be mostly right is still not
research, because you can't tell the mostly-right parts from the wrong one
without doing the verification yourself, which is what happened.

## 1. Verified pricing (primary sources only, fetched 2026-07-16)

Not aggregator/SEO sites (`aipricing.guru`, `benchlm.ai`, etc. — these showed up
first in search and are the kind of programmatically-generated content that
looks authoritative and often isn't). Pulled directly from each vendor's own
docs:

| Model | Input | Cached input | Output | Batch (input/output) | Source |
|---|---:|---:|---:|---:|---|
| GPT-5.6 Sol | $5.00 | $0.50 | $30.00 | $2.50 / $15.00 | [OpenAI](https://developers.openai.com/api/docs/pricing) |
| GPT-5.6 Terra | $2.50 | $0.25 | $15.00 | $1.25 / $7.50 | [OpenAI](https://developers.openai.com/api/docs/pricing) |
| GPT-5.6 Luna | $1.00 | $0.10 | $6.00 | $0.50 / $3.00 | [OpenAI](https://developers.openai.com/api/docs/pricing) |
| GPT-5.4-mini | $0.75 | $0.075 | $4.50 | $0.375 / $2.25 | [OpenAI](https://developers.openai.com/api/docs/pricing) |
| GPT-5.4-nano | $0.20 | $0.02 | $1.25 | $0.10 / $0.625 | [OpenAI](https://developers.openai.com/api/docs/pricing) |
| Gemini 3.5 Flash | $1.50 | — | $9.00 | not published | [Google](https://ai.google.dev/gemini-api/docs/pricing) |
| Gemini 2.5 Flash | $0.30 | $0.03 (cache)† | $2.50 | $0.15 / $1.25 | [Google](https://ai.google.dev/gemini-api/docs/pricing) |
| Gemini 2.5 Flash-Lite | $0.10 | — | $0.40 | not checked | [Google](https://ai.google.dev/gemini-api/docs/pricing) |
| Claude Fable 5 | $10.00 | $1.00 | $50.00 | $5.00 / $25.00 | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |
| Claude Sonnet 5 (thru 2026-08-31) | $2.00 | $0.20 | $10.00 | $1.00 / $5.00 | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) — rises to $3/$15 Sept 1 |
| Claude Haiku 4.5 | $1.00 | $0.10 | $5.00 | $0.50 / $2.50 | [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing) |

† Gemini's context caching bills differently (a small per-token caching charge
plus $1/M-tokens/hour storage), approximated in our model as the batch rate —
see `cost.py` comments; re-verify precisely before a real spend.

**GPT-5.6 availability**: previewed June 26 2026 gated to ~20 partners, went to
**general availability July 9 2026** — one week before this research. It's
usable now, but *because* it's a week old, no third-party benchmark has run it
on extraction tasks yet (see §2). [Sources on the gating: explainx.ai](https://www.explainx.ai/blog/when-will-gpt-5-6-sol-terra-luna-be-available-everyone-2026), [digitalapplied.com](https://www.digitalapplied.com/blog/gpt-5-6-sol-terra-luna-public-ga).

**Excluded from the running: Fable 5, Sol, Gemini 3.5 Flash's premium tier.**
Not because they're bad — because at 4–10x the mid-tier price with zero
task-specific evidence they extract more accurately, they cannot be the answer
to a sub-$0.01/image target regardless of how good they are.

## 2. Precision: what evidence actually exists (with honest limits)

**No benchmark exists for our specific task** — historical Spanish/Portuguese
sacramental-register normalization + structured extraction. The closest
academic work is OCR post-correction research (HIPE-OCRepair 2026; [arXiv
2502.01205](https://arxiv.org/html/2502.01205v1), titled, tellingly, "No Free
Lunches"), which evaluates open models (Llama, Gemma, Mixtral) on historical
Spanish newspaper text, not the commercial APIs in this comparison, and
explicitly cautions against assuming any model is uniformly good at
distinguishing genuine OCR error from archaic-but-correct spelling — exactly
the judgment call our normalization prompt asks for.

The closest *available* signal is general JSON-extraction benchmarking. From
[one independently-run benchmark](https://ianlpaterson.com/blog/llm-benchmark-2026-38-actual-tasks-15-models-for-2-29/)
(a single source — not peer-reviewed, treat as a data point, not a verdict):

| Model | Extraction quality | Note |
|---|---:|---|
| Gemini 2.5 Flash | 97.1% | best score among directly-comparable vendor models |
| Claude Haiku 4.5 | 95.9% | called out specifically for instruction-following reliability |
| GPT-5.4-nano | — | "single-digit accuracy gap" vs. premium tiers, ~30x cheaper |

GPT-5.6 (any tier) and Gemini 3.5 Flash have **no published extraction
benchmark** as of this writing — they're too new. Claude Sonnet 5 wasn't in
that particular benchmark either, though it's a strictly more capable model
than Haiku at a similar price band (with introductory pricing through Aug 31).

**Why this matters for model choice, not just cost:** our task is closer to
disciplined instruction-following (a long list of specific rules — strip this,
expand that, never invent) than open-ended reasoning. That favors models that
score well on *reliability at following detailed formatting rules* over raw
frontier capability — which is exactly the axis Haiku 4.5 was called out on,
and is consistent with Gemini 2.5 Flash's benchmark lead being on a
JSON-extraction task specifically rather than a reasoning one.

## 3. Real cost, computed through our own pipeline (not hypothetical)

Using `ssda_nlp_tools/cost.py` with the verified prices above and this
project's actual measured token footprint (system prompt, 15 real few-shot
examples, real entry sizes), at **full quality — 15 shots, no cuts** — batched
10/call, cached, normalization folded in:

| Model | $/image (total) | Whole corpus (72,194 images) | Under $0.01? |
|---|---:|---:|:---:|
| **Gemini 2.5 Flash** | **$0.0079** | **$573** | ✅ |
| GPT-5.4-nano | $0.0048 | $347 | ✅ |
| Claude Haiku 4.5 | $0.0135 | $978 | ❌ |
| GPT-5.6 Luna | $0.0157 | $1,133 | ❌ |
| Claude Sonnet 5 | $0.0251 | $1,816 | ❌ |
| GPT-5.6 Terra | $0.0363 | $2,621 | ❌ |

**This revises an earlier number in this project.** A cost report I wrote
before doing this research quoted "~$52 for the whole corpus," using a pricing
table explicitly labeled "representative early-2026 list rates" — i.e.
placeholder numbers, not verified ones. Real July-2026 prices are higher, so
the real achievable number is **$573 (Gemini 2.5 Flash) or $347 (GPT-5.4-nano)**,
not $52. Both are still comfortably under the $0.01/image target and far under
the $0.05 cap — the target is still met, just not by as wide a margin as the
placeholder number implied. Worth flagging to Daniel since that $52 figure may
already have been mentioned to him.

Corpus size note: 72,194 is the image count implied by our own measured
entries-per-image ratio (2.45) applied to the 176,876 entries we've actually
segmented — recompute from your real volume count if it's not 750k.

## 4. Recommendation

**Primary: Gemini 2.5 Flash.** Best directly-relevant benchmark score found,
comfortably under budget at full 15-shot quality (no accuracy-for-cost
tradeoff needed), and it's the same family already used for transcription in
our cost model, so one vendor relationship covers both stages.

**Strong alternative: GPT-5.4-nano.** Cheapest of the viable set and the
"single-digit accuracy gap" framing is a genuine signal, but it's a *relative*
claim without an absolute number for extraction specifically — worth the live
test in §5 before committing.

**If Anthropic is preferred for other reasons** (e.g. already integrated,
team familiarity): **Claude Haiku 4.5**, not Sonnet 5 — Haiku's specific
instruction-following callout matches this task better than Sonnet's general
capability edge, and it's roughly half Sonnet's cost. Both currently land
over $0.01/image at 15 shots in our model; getting under would mean the
shot-cut this project has already found costs accuracy (§ the bake-off in
`model_agreement_0013_0023.md`), so I would not recommend forcing an Anthropic
model under budget that way.

**Do not use Fable 5, GPT-5.6 Sol, or Gemini 3.5 Flash for bulk extraction** —
no evidence they extract this kind of record more accurately, at a price that
would blow the budget regardless.

## 5. What this report can't tell you — and what would

No benchmark here was run *on our actual data*. The honest next step is
`run_live_test.py` (already built, already dry-run verified against volume
239746): point it at Gemini 2.5 Flash and GPT-5.4-nano with a small
`--max-batches` and a hard `--max-usd` cap, and compare the **actual** returned
JSON against a few of Daniel's hand-corrected gold entries. That converts
"best available signal" into a real answer for this specific task, for a few
cents.

## Reproduce

```bash
python run_cost.py --target 0.01 --model gemini-2.5-flash --corpus 72194
python run_cost.py --target 0.01 --model gpt-5.4-nano --corpus 72194
python run_live_test.py out_batches/239746.batches.jsonl --max-batches 3   # dry run
```
