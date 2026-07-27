# Runbook — what to run next, and the one trap to avoid

State as of 2026-07-27. Everything below is offline/$0 unless marked **PAID**.

## 1. When the 24-request job lands (repairs + vocab test)

The monitor should do this automatically. To check, or to do it by hand:

```bash
python assemble_corpus.py      # offline: repairs -> corpus, vocabtest -> separate file
python vocab_ab_report.py      # offline: the age/ethnicity verdict
```

**Expected if it worked:** `201991` 2,021/2,021 and `29597` 781/781 in
`CORPUS_SUMMARY.json` (the 25 gaps filled), plus a verdict line.

Read the verdict literally:
- `PROMPT WORKS` — both age *and* ethnicity improved on real values. Go to §2.
- `PARTIAL` — one dimension improved, the other was flat or **UNMEASURED**.
  UNMEASURED means the new extraction emitted no values for that field; it is
  not evidence of improvement. Inspect the off-vocab list before spending.
- `NO IMPROVEMENT` / `REGRESSION` — do not spend; diagnose.
- Ignore **phenotype** either way. On 701054 every miss is `preto`/`preta`
  (absent from vocab.json) or a feminine form of a listed masculine entry. That
  is a vocabulary gap awaiting Daniel's call — no prompt can fix it.

## 2. **PAID** — the full re-extraction (~$15), only if §1 says it is justified

Batches are already staged with the vocabulary-aware prompt in
`production/batches_v2/` (5 volumes, 527 calls, ~$15.09 Batch API).

> ### ⚠️ The trap: send v2 output to a SEPARATE directory
> v2 reuses the same `<vol>-bNNNN` custom_ids as the delivered run, so its
> records carry the **same entry IDs** as the 17 existing `*.output.jsonl` files
> in `production/luna_live/`. Assembling both together makes every record a
> duplicate; the de-duplicator keeps whichever file sorts first and flags the
> rest. You would not lose data, but the corpus would be an unpredictable mix of
> old and new extractions.
>
> This is the same collision class that was fixed for the vocab test — there it
> is handled in code, here it must be handled by **choosing the output path**.

> ### ⚠️⚠️ Second trap, and it costs money: **the cap is per-directory**
> `run_luna_production.py` derives its ledger as `--outdir/spend_ledger.json`.
> So the isolation above has a side effect: a fresh `--outdir` starts a **fresh
> ledger with a fresh $20 cap**, and it will cheerfully report
> `available $20.000000` even though $12.01 is already committed in
> `luna_live`. Nothing sums the two for you.
>
> Total real exposure would be **$12.01 + up to $20 = $32**, while every
> individual run looks compliant.
>
> **Therefore: pass an explicit `--cap-usd` equal to the budget you actually
> intend for the new directory**, and add the ledgers yourself when reporting
> spend.

```bash
# Per volume, into a fresh directory, with an EXPLICIT cap for that directory.
# --take is the request count for that volume (176899 = 109; check the file).
python run_luna_production.py production/batches_v2/176899.batches.jsonl \
    --outdir production/luna_v2 --cap-usd 16.00 --take 109 --confirm
# ... repeat per volume; the luna_v2 ledger accumulates across them ...

# assemble the NEW corpus from the NEW directory only
python assemble_corpus.py --live production/luna_v2 --corpus production/corpus
```

A full v2 run is ~$15.09, so `--cap-usd 16.00` on the new directory gives a
little headroom without opening an unbounded second budget. Combined ceiling
then reads **$20 (luna_live) + $16 (luna_v2)** — a deliberate number, not an
accident.

Keep `production/luna_live/` intact until the v2 corpus is checked — it is the
current delivered dataset and the only copy of the baseline extraction.

## 3. Ledger — one per output directory, NOT one global

Each `--outdir` owns its own `spend_ledger.json` and its own cap. There is no
global total; you must add them.

| ledger | cap | committed |
|---|---|---|
| `production/luna_live/spend_ledger.json` | $20 | $11.051032 + $0.96 reserved = **$12.011032** (headroom $7.99) |
| `production/luna_v2/spend_ledger.json` | *set by `--cap-usd`* | does not exist until a v2 run is submitted |

A full v2 run (~$15.09) does **not** fit in `luna_live`'s remaining $7.99. That
is a genuine signal, not an obstacle to route around: re-extracting the whole
corpus is a real second spend and should be an explicit decision with its own
stated cap.

## 4. Open, needs Daniel

- **Phenotype vocabulary gap**: add `preto`/`preta` and feminine forms, or fold
  gender before checking? Raised in `DANIEL_REPLY_20260727.md`.
- **Reply not yet sent** — draft is in `production/luna_live/`.
- 108 fallback records, 16 re-transcribe pages, 3952 admin material: separate
  tracks, unchanged.

## 5. Untracked on purpose

`run_sonnet_cached_batch.py`, `submit_gemini_batch.py`, `submit_sonnet_batch.py`
submit paid jobs with **no `--confirm` guard and no ledger**. They predate the
guarded runner and are deliberately left untracked. Do not commit them without
adding the guards first.
