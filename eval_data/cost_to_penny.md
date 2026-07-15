# Getting transcription + normalization to ≤ $0.01 / image

Generated offline by `ssda_nlp_tools.cost` (no API calls). Token counts are
**measured from the repo's own files** — `extract.py`'s system prompt,
`instructions.json`, `training_data.json`, and a real volume — not assumed.
Prices are a configurable table of representative early-2026 list rates
(`--pricing prices.json` to override); the **levers and their relative sizes are
robust to the exact prices**, which is what matters for the decision.

Reproduce:
```bash
python run_cost.py --target 0.01            # cost model + optimizer + waterfall
python run_batch.py Sample_output/Generated_0013_0023_4o_prompt_V2.json \
                    --instructions instructions.json --batch 10   # the token saving, live
```

## Measured footprint (per the real files)

| component | tokens | note |
|---|---:|---|
| extraction system prompt | ~1,250 | `EXTRACTION_SYSTEM_PROMPT` |
| normalization system prompt | ~490 | separate LLM pass today |
| instructions.json | ~220 | |
| one few-shot example | ~740 | ×15 shots = **~11k static tokens** |
| per-entry input | ~185 | normalized transcription |
| per-entry output | ~690 | the `data` JSON |
| entries per image | **2.45** | 27 entries / 11 pages |
| whole-page transcription output | ~580 | Gemini/Archivault |

**The diagnosis in one line:** the ~11k-token few-shot prefix is billed **once
per entry** today, and normalization is a **second full LLM pass**. That is the
entire cost problem.

## The lever waterfall (extraction model = gpt-4o-mini)

| step | trans+norm/img | total/img | marginal save |
|---|---:|---:|---:|
| baseline: per-entry call, 15 shots, no cache, separate norm | $0.0051 | $0.0108 | — |
| **+ prompt caching** (shared prefix cached across the volume) | $0.0029 | $0.0063 | −$0.0045 |
| **+ batch 10 entries/call** (amortize the prefix 10×) | $0.0010 | $0.0023 | −$0.0040 |
| **+ fold normalization into extraction** (2 passes → 1) | $0.0007 | $0.0020 | −$0.0003 |
| + drop to 5 shots *(accuracy-sensitive — skip this one)* | $0.0007 | $0.0019 | −$0.0001 |

**Quality-preserving endpoint: $0.0020/image total** (vs $0.0108) — **5× cheaper**,
and transcription + normalization is **$0.0007**, comfortably under the $0.01 goal.
Across 750k images: **$8,085 → $1,500** (saves ~$6,600) with *no* accuracy change.

Same levers on a flagship model (gpt-4o): **$0.173 → $0.027/image**, i.e.
**$130k → $20k** over the corpus. The levers matter more the pricier the model.

## The recipe (do these three; skip the fourth)

1. **Prompt caching.** Put every static token (system prompt, instructions,
   few-shots) first and byte-identical across calls; only the entries vary at the
   tail. OpenAI caches automatically ≥1024-token prefixes; Anthropic/Gemini via
   explicit cache. → biggest single win, **zero quality cost**.
2. **Batch 10–20 entries per call.** Pay the prefix once per batch, not per
   entry. → second-biggest win, **zero quality cost**.
3. **Fold normalization into the extraction call.** Ask for `normalized` *and*
   `data` in one response — eliminates a whole LLM pass. The driver already has a
   `LLM_REPAIR_RETURNS_NORMALIZED` hook for exactly this.
4. **Do NOT cut few-shots to save money.** The waterfall shows it saves ~$0.0001
   while the bake-off (`model_agreement_0013_0023.md`) shows it measurably drops
   people-F1. Caching already makes the shots nearly free after the first call.

Model tier is a separate, orthogonal lever: mid-tier (gpt-4o-mini / gemini-flash)
already clears the target; nano/flash-lite clear it with headroom. Choose the tier
on **accuracy** (use the eval harness), not on cost — cost is solved by 1–3.

## Verified implementation

`ssda_nlp_tools/batch_extract.py` implements levers 1–3 and is verified offline
on the real chunk:

* **cache-ordered** — the static prefix is byte-identical across batches (test).
* **round-trips** — a batch response parses back to correct per-entry `data` (test).
* **8.06× fewer input tokens** on the real 27-entry chunk (392,202 → 48,643), and
  **27 separate normalization calls eliminated**.
* `run_batch.py --emit` writes a ready-to-send `messages` array (see
  `eval_data/messages_batch1_example.json`).

The one number I can't measure offline is Gemini's per-page **image** token count
(transcription input). Across 2–10 tiles/page it moves transcription only
$0.0002 → $0.0004/image — immaterial to the target either way.
