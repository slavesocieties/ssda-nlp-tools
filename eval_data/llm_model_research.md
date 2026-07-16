# Which LLM for extraction: cost vs. precision (updated 2026-07-16)

Second pass on this, after Ronak's follow-up asked for two real fixes and a
change of framing:

1. **Model coverage** — properly evaluate Gemini 3.5 Flash (not lump it in with
   the flagship-tier exclusion) and GPT-5.6 Luna, and account for Gemini's
   Batch API pricing.
2. **Claude Sonnet 5 prompt caching** — the first pass under-modeled it. This
   pass found and fixed a real bug: the cost model never actually implemented
   the vendor's *async Batch API* discount as a distinct thing from prompt
   caching, so every model — Claude included — was being costed without its
   biggest available discount. Fixed below; it changes the answer.
3. **Decision rule** — this is a **one-time run**. Ronak's stated rule: paying
   up to ~1¢/image for a materially better result (his framing: "99% good")
   beats saving a fraction of a cent for a worse one ("80% good"). The first
   pass optimized for *cheapest option that clears a quality floor*; this pass
   adds a **quality-first mode** that instead maximizes quality *within* the
   budget, which is what a single production run actually calls for.

Prior report for context: `git log` / previous version of this file. The
headline number from that pass — Gemini 2.5 Flash the standout, ~$573 for the
corpus — **still holds**, but the margin over Claude is much smaller than
reported, because of the bug in point 2.

## 1. The bug: batch API discount was never modeled as its own thing

Every vendor offers **two independent discounts** that combine differently:

- **Prompt caching** — repeated static content (our ~14k-token instruction +
  few-shot prefix) bills at a fraction of input price on cache hits.
- **The vendor's async Batch API** — accept a 24-hour turnaround window, get a
  flat discount on *every* token, cache or not.

The first pass's cost model only ever implemented prompt caching. It had no
representation of the Batch API discount at all — meaning every "with caching"
number for every model, Claude included, was missing its single biggest lever.
Fixed now, with the two discounts modeled as genuinely independent and
**stacked per-vendor using the rates each vendor states for the combination**,
not a guessed product of the two:

| Vendor | Batch alone | Cached + Batch together | Source |
|---|---:|---:|---|
| Anthropic | 50% off | **5%** of standard (their own words: "as little as 5%") | [platform.claude.com](https://platform.claude.com/docs/en/about-claude/pricing) |
| OpenAI | 50% off | **25%** of standard (their own reported example: GPT-5.4 cached+batched input = $0.625/M vs $2.50 base) | search-aggregated from OpenAI's public statements — see caveat below |
| Google | 50% off (confirmed for both Flash models) | not independently verified — modeled conservatively, flagged in every report | [ai.google.dev](https://ai.google.dev/gemini-api/docs/pricing) |

Note the OpenAI figure (25%) is *not* the naive 5% you'd get multiplying
50% × 10% — worth knowing because assuming naive multiplication would have
made OpenAI models look cheaper than they actually are. The OpenAI number came
from search-result synthesis rather than a page I fetched and read myself
line-by-line the way I did for the three primary pricing pages, so it carries
slightly less certainty than the Anthropic figure — flagged accordingly.

**This changes Claude's standing materially.** With the discount now modeled
correctly:

| Model | Old number (bug) | Corrected | Change |
|---|---:|---:|---|
| Claude Haiku 4.5 | $0.0135/image (over budget) | **$0.0059/image (under budget)** | was wrongly excluded from budget-fitting options |
| Claude Sonnet 5 | $0.0251/image | **$0.0117/image** | still just over $0.01, much closer than reported |

## 2. Gemini 3.5 Flash and GPT-5.6 Luna, properly evaluated (not excluded)

The first pass lumped Gemini 3.5 Flash in with the flagship/reasoning-tier
exclusion. That was a miscategorization — at $1.50/$9.00 it's priced like a
mid-tier workhorse (nowhere near Fable 5's $10/$50 or GPT-5.6 Sol's $5/$30),
so it belongs in the real comparison, not excluded on a cost-tier assumption.
Doing that properly:

| Model | $/image (best recipe: full 15 shots, cached, batch API, batched 20/call) | Whole corpus (72,194 img) | Extraction benchmark? |
|---|---:|---:|---|
| Gemini 2.5 Flash | $0.0030 | $216 | ✅ 97.1% — the best sourced score found |
| GPT-5.4-nano | $0.0017 | $121 | partial — "single-digit gap vs premium," no absolute number |
| GPT-5.4-mini | $0.0055 | $399 | none found |
| Claude Haiku 4.5 | $0.0059 | $427 | ✅ 95.9% + instruction-following callout |
| GPT-5.6-luna | $0.0073 | $526 | none — GA one week before this research |
| Gemini 3.5 Flash | $0.0104 | $750 | **none** — too new (GA 2026-05-19, no independent extraction benchmark surfaced) |
| Claude Sonnet 5 | $0.0117 | $841 | none in the benchmark found (Haiku was tested, Sonnet wasn't) |
| GPT-5.6-terra | $0.0179 | $1,295 | none — GA 2026-07-09, one week old |
| Gemini 2.5 Flash-Lite | $0.0007 | $48 | none — assumed weakest, no evidence either way |

**Honest answer on "go Gemini 3.5 Flash":** I did evaluate it properly this
time, and the finding doesn't support switching to it — it costs **3.5x more
than Gemini 2.5 Flash** ($0.0104 vs $0.0030/image) with **zero evidence it
extracts more accurately** (no benchmark exists for it yet; it's two months
old). If a benchmark surfaces showing it's meaningfully better, that would
change the answer — today, nothing supports paying the premium.

## 3. Quality-first recommendation (your actual decision rule)

You said: up to ~1¢/image is fine for a materially better result, since this
runs once. That's a different question than "cheapest that clears a bar" — so
`cost.py` now has a dedicated mode (`optimize_for_quality`, `--quality-first`
on the CLI) that **never cuts shots** (the accuracy cost of doing that was
already measured and isn't worth it at this volume) and picks the
**highest-quality option that still fits the budget**, not the cheapest one.

**Primary: Gemini 2.5 Flash.** It wins on both axes now, not just cost — best
sourced quality evidence (97.1%) *and*, once Batch API is modeled correctly,
also the cheapest of the evidenced-strong options ($0.0030/image, $216 for the
whole corpus). There's no real tradeoff to make here; take it.

**If there's a non-cost reason to prefer Anthropic** (vendor relationship,
data handling terms, team familiarity): **Claude Haiku 4.5**, now genuinely
available at $0.0059/image with correct batch+cache stacking, and specifically
called out in the benchmark for instruction-following reliability — a good
match for a task that's mostly "follow this list of normalization rules
precisely," not open-ended reasoning. This is a real reversal from the first
pass, which wrongly showed it over budget.

**Claude Sonnet 5** is a genuinely more capable model than Haiku and, with
full discount stacking, lands at $0.0117/image — technically over the strict
$0.01 line, but close enough that if "up to a cent" has any flex, it's worth
a real test (§4) rather than ruling out on a ~15% overage with no benchmark
data either way.

**Do not use**: Fable 5, GPT-5.6 Sol (flagship tier, no evidence of an
accuracy edge worth 4-10x the cost), Gemini 3.5 Flash (costs more than 2.5
Flash with no accuracy evidence), or GPT-5.6 Terra (real and GA, but at
$0.0179/image with zero benchmark data, it's paying a premium on faith).

## 4. What would actually settle this

Every number above is projection or one secondhand benchmark — none of it is
a measurement on *this* project's actual text. `run_live_test.py` (already
built, dry-run verified) is exactly the tool for closing that gap: point it at
Gemini 2.5 Flash and Claude Haiku 4.5 with a small `--max-batches` and a hard
`--max-usd`, and diff the real returned JSON against a few of Daniel's
hand-corrected gold entries. A few cents of real spend beats another round of
projection.

## 5. Output format: faithful and normalized, both kept

Confirmed decision (2026-07-16): the pipeline keeps **both** the faithful text
(exactly what Archivault/the segmenter produced — free, auditable, no
model-invented content) and the LLM's normalized text, side by side, in every
final record — neither replaces the other.
`ssda_nlp_tools.batch_extract.merge_with_faithful()` does this join; entries
the model doesn't return (dropped, still pending, etc.) are still present in
the output with `text_normalized`/`data` left `null` rather than silently
disappearing, consistent with how partial records are handled elsewhere in
this project. `run_live_test.py` now writes this merged form (`"records"` in
its output JSON) alongside the raw usage/cost data.

## Reproduce

```bash
python run_cost.py --target 0.01 --quality-first          # the recommendation above
python run_cost.py --target 0.01 --model gemini-2.5-flash --corpus 72194
python run_live_test.py out_batches/239746.batches.jsonl --max-batches 3   # dry run
python -m pytest tests -q                                  # incl. the stacking regression tests
```
