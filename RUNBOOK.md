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

> ### ⚠️⚠️ Second trap, and it costs money: **the ledger must be shared**
> A separate output directory is necessary for data isolation, but it must not
> create a separate budget. The guarded runner now **refuses** any non-default
> `--outdir` unless `--ledger-path` is supplied. Point it at the live ledger so
> the existing $20 cap remains cumulative.
>
> A re-extraction also needs a distinct `--run-id`: the source compact files
> reuse `<volume>-bNNNN` custom IDs, and the shared ledger correctly treats
> those original IDs as already sent. `--run-id v2` namespaces only provider
> request IDs; source entry IDs remain unchanged and auditable.

```bash
# Per volume, into a fresh directory while retaining the ONE cumulative ledger.
# Run without --confirm first. --take is request count (176899 = 109).
python run_luna_production.py production/batches_v2/176899.batches.jsonl \
    --outdir production/luna_v2 \
    --ledger-path production/luna_live/spend_ledger.json \
    --run-id v2 --cap-usd 20.00 --take 109

# assemble the NEW corpus from the NEW directory only
python assemble_corpus.py --live production/luna_v2 --corpus production/corpus
```

A full v2 run is ~$15.09, which does **not** fit inside the current $7.99
global headroom. The command above must refuse once its reservation would cross
the live ledger's $20 cap. Raising that total cap is a separate, explicit user
approval; it is never achieved by creating a second ledger.

Keep `production/luna_live/` intact until the v2 corpus is checked — it is the
current delivered dataset and the only copy of the baseline extraction.

## 3. Ledger — one cumulative budget, separate artifact directories

`production/luna_live/spend_ledger.json` is the cumulative production ledger.
All isolated re-extractions must reference it with `--ledger-path`; their
receipts and downloaded outputs still live under their own `--outdir`.

| ledger | cap | committed |
|---|---|---|
| `production/luna_live/spend_ledger.json` | $20 | $11.051032 + $0.96 reserved = **$12.011032** (headroom $7.99) |
| `production/luna_v2/` | no independent ledger | isolated v2 receipts, outputs, and assembled corpus only |

A full v2 run (~$15.09) does **not** fit in `luna_live`'s remaining $7.99. That
is a genuine signal, not an obstacle to route around: re-extracting the whole
corpus needs a new, explicitly approved global cap.

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
