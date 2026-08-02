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

## 3a. Merging is now its own stage (2026-07-29)

Daniel: "handle merging completely separately from extraction." Extraction is
paid and settled; merging is free and its rules are still moving. Keeping them
fused made every merge experiment look like it needed a re-extraction.

```powershell
# $0, minutes, reads delivered extraction output and never writes to it
python run_merge.py --tag v3
python run_merge.py --tag loose --no-surname-tiers      # A/B the Llopiz tiers
python run_merge.py --tag v4 --constraints labels.json  # feed review back in

# $0: stratified pair sample + the 0/25/50/75/100 labelling page
python build_training_sample.py --tag core --size 2000
```

**Trap.** `disambiguate_volume(pair_log=...)` scores tens of millions of pairs.
Pass `collect_review=False` alongside it or the function also materialises a
~1.1M-entry review queue with evidence cards attached, which is gigabytes built
to be thrown away — it took the first training-sample run to 687 MB and climbing
before the guard existed.

**Trap.** The pair log must be given a bounded sink (`StratifiedReservoir`), not
a plain list, for the same reason.

## 3b. Merging: what NOT to try next (2026-07-29)

Daniel: "No people should be merged strictly based on name correspondence" and
"these last names are not special cases." Nothing merges on a name now; a merge
needs a counted combination of date overlap, same-named relation, matching
qualities, or an identifying relation. Shared register counts for nothing --
everyone in a volume has it, and treating it as evidence is what let a single
weak signal clear a score threshold.

**Do not try to infer inherited vs devotional "de la Cruz" from surrounding
relationships.** It sounds tractable and is not: the parent of a "María de la
Cruz" is often *also* recorded "de la Cruz" precisely because the epithet was
applied to the whole household, so the corroboration signal and the thing being
tested are entangled. `name_epithets.json` survives only because reading "N."
(nomen nescio) as a surname is factually wrong, not because it is load-bearing.

**Do not keep tuning thresholds.** Measured: after removing the exemption,
"María del Rosario" still holds 31 mentions and "María de la Concepción" 45,
because two different women of that name, both parda, both free, baptised in the
same decade, genuinely have two matching signals. No threshold separates them.
That is the boundary where rules stop and the trained model starts, which is
Daniel's position. The next lever is labelled data, not another rule.

## 4. Open, needs Daniel

- **701162** — no Archivault transcription exists; the 232-volume set does not
  include it and Drive holds only page images. Needs transcription before it can
  be segmented. 701179 is done and free (697 entries, 1 partial).
- **The two Llopiz bridges** — `llepiz~yepez` (11 mentions) and `llopez~lopez`
  (40) survive the tiered guard because `phonetic_fold` maps both `ll` and `y`
  to `i`. Loosening yeísmo affects every name in the corpus, so it is his call.
  See `production/luna_v3/DANIEL_2026-07-29_IMPLEMENTED.md` §2.
- **Eight ethnicity terms added against our judgement** and flagged in
  `vocab_extensions.json` (`moreno` is a phenotype value, `Agustino` a religious
  order, `Cimarrón` a status, `casta` and `nación no conocida` placeholders,
  three that read as surnames).
- **The ~10% arithmetic** — a literal 10% of the pair queue is ~70,000
  decisions, ~195 hours. The stratified sample is the tractable alternative.
- 108 fallback records, 16 re-transcribe pages, 3952 admin material: separate
  tracks, unchanged.

## 4a. Resolved by Daniel, 2026-07-29

- Ethnicity: all 71 queued terms added; conformance 94.8% → **100.0%**.
- Llopiz/Llopis merge on reasonable context, Llepiz on real corroboration,
  Llepico only on very clear context — encoded as `SURNAME_TIERS`.
- Review will be **pairwise**, labelled 0/25/50/75/100, and that data trains the
  merge model. Only 0% and 100% become hard constraints.
- Next volumes: 701162 and 701179.

## 5. Untracked on purpose

`run_sonnet_cached_batch.py`, `submit_gemini_batch.py`, `submit_sonnet_batch.py`
submit paid jobs with **no `--confirm` guard and no ledger**. They predate the
guarded runner and are deliberately left untracked. Do not commit them without
adding the guards first.

## 6. Transcription: what exists, and what is missing (2026-07-31)

Read from source, not assumed. `ssda-htr` **does not transcribe**. `driver.py`
pulls a page, pools/resizes to 960x1280, runs `layout_analyze` to find blocks
(= entries, the thing Daniel's masks are ground truth for), deskews each, cuts
it into LINE images, and PUTs them to a bucket named `ssda-htr-training`. It
returns a count. No text is produced anywhere in the repo.

So the line-cutting half of an HTR system exists and has been run; the
recogniser does not exist and the crops appear to be unlabelled. All actual
transcription is Gemini, whole-page, in the Archivault backend Lambda.

**Blocking fact for any HTR training:** the 232 volumes with transcriptions and
the 6 volumes whose images we can reach are DISJOINT (`ssda-production-jpgs`
returns AccessDenied). Usable pairs today are ~1,600, not 62,320. That is an
access problem, not a data problem.

**SECURITY, reported to Daniel, not ours to fix:** `driver.py` uploads each crop
with a plain unauthenticated `requests.put` to a public API Gateway URL. No key,
token or signature anywhere in the repo. If the gateway does not enforce auth,
that training bucket is writable by anyone who knows the URL, which is a
data-poisoning surface rather than a mere storage leak. NOT tested from here --
probing their endpoint is not ours to do. Our side is clean: every outbound call
in this repo carries authentication.

**Do not train a recogniser on Gemini output and expect to beat Gemini.** The
student inherits the teacher's errors, including the Dezembro-for-Novembro miss
on 701157-0056. Human-corrected pages are worth more per page than machine ones,
for measurement as well as training.

## 7. Hand transcriptions: the first real accuracy measurement (2026-07-31)

Daniel: the current pre-summer infrastructure is `slavesocieties/openai`, and it
carries hand transcriptions in `json/`. Nine volumes, 3,452 entries. Three
(1795, 15834, 419324) are also in the 232-volume Archivault set.

```bash
python run_manual_gold.py     # $0, offline, needs ../ssda-openai and ../transcriptions
```

Result over 335 pages / 421,756 human characters: substitution 6.31%, deletion
9.06%, insertion 15.27%, median page similarity 0.891. Read the three rates
SEPARATELY. Insertion is mostly scope (human transcribed entries, machine
transcribed whole folios with marginalia and stamps), so it is not error.
Substitution is the quality signal.

**Trap that already cost one wrong set of figures: page alignment DRIFTS.** In
15834 human pages 30-69 sit at machine offset +0, 70-189 at +1, 190 onward back
at +0. A constant-offset check reports +0 as globally best, which is true on
average and wrong for 129 pages, and both texts are well-formed register prose
so nothing downstream complains. `offset_map` follows the drift using the modal
offset of NEIGHBOURING pages. Never align a page by its own best score: that
maximises similarity by construction and inflates every accuracy number.

**Trap: exclude hard failures before averaging.** 12 of these pages contain only
`[TRANSCRIPTION FAILED: ...]`. Averaging their 100% deletion into an accuracy
figure measures availability, not quality.

**Corpus-wide finding: 1,281 of 62,320 pages (2.06%) across 184 of 232 volumes
carry that marker instead of a transcription**, in four variants; 266286 is 34%
failed. `transcription_integrity.check_page` now catches it exactly (a literal
tooling marker, so no false positives, unlike the three heuristics that were
cut). Our delivered corpus is clean: 701157's four are covers/end matter.

## 8. The 1,000-pair labelling set (2026-07-31)

```bash
python build_training_sample.py --size 1000 --tag daniel1k
```

444 strata, all covered. Singletons fall 288 (size 600) -> 88 (800) -> 37 (1000)
and then stay at 37, because those 37 case types occur exactly once in 7.3M
pairs. So 1,000 is the knee; more budget only deepens cells that already have
depth. Daniel asked for ~1k and ~1k is right.

`info_bucket` is the axis he asked for and we lacked ("range of pieces of
identifying information"), bucketed on the POORER side because that bounds the
decision. Rich pairs went 1.7% -> 12.5% like-for-like. Do NOT quote "19 of
2,000" as the before-figure; that is a stricter statistic and counted `context`.

The scoring pass is cached (`_reservoir.pkl`, ~506s). `--rescore` to redo.

## 9. The segmenter fix does NOT justify re-extracting the delivered corpus

Measured, because "we improved the segmenter" normally implies "re-extract",
which is ~$15 and invalidates the delivered dataset.

Re-segmenting all five delivered volumes from `_task2/drive_ready/<vol>/<vol>.json`
with the fixed segmenter changes the entry count by **+1 out of 5,343 (0.02%)**,
and the one change is in 29597.

The reason is orthographic era, not luck. The "mill" and "Marte" fixes target
18th-century scribal habits:

| volume | mil | mill | era |
|---|---|---|---|
| 176899 | 1,050 | 0 | 19th |
| 201991 | 2,200 | 4 | 19th |
| 29597 | 756 | 11 | **18th** |
| 375062 | 1,131 | 0 | 19th |
| 701054 | 679 | 0 | 19th |
| 15834 (gold) | 257 | **369** | 18th |

Our delivered corpus is overwhelmingly 19th-century, where "mil" is standard.
15834 is 18th-century and writes "mill" more often than "mil". 29597 is our only
18th-century volume and is exactly where the +1 landed.

So: keep the fix (it is real and free), do NOT re-extract, and expect it to
matter for older volumes as they arrive.

## 10. Two silent-subtraction bugs, both found by RUNNING things (2026-07-31)

Neither raised. Both removed data.

**Re-assembly dropped 160 paid records.** Repair requests are addressed to an
entry, so their custom_id is `v3-repair1-176899-0236-B-01` -- volume followed by
a page number, not by `-b0`. `_VOL_RE` matched none of them and unmapped ids are
skipped silently, so `assemble_corpus.py` took the corpus from 5,226 to 5,066
with every volume downgraded to PARTIAL. Second time this mapping has discarded
delivered work (see §the whitelist note). Fixed + tested against every custom_id
shape actually present.

**Withdrawn records came back.** `withdraw_records.py` edits the materialized
files; assembly rebuilds them from source. Assembly now re-applies
`withdrawn_records.json` every run. A withdrawal that lasts until the next
rebuild is not a withdrawal.

**Trap for anyone re-running assembly:** always check the record total against
CORPUS_SUMMARY afterwards. Both bugs above are invisible in the exit code.

## 11. Markdown-table transcriptions (2026-07-31)

3,622 of 62,320 pages (5.81%, 123 of 232 volumes) are transcribed as markdown
tables, because the registers are physically two-column and Gemini renders that
faithfully. 439941 is 85.8% tables. `detect_page_type` read the pipes as an
INDEX page and skipped them whole -- zero entries each.

`strip_markdown_table` flattens them before classification. Benchmark: entries
found 96.4% -> 98.2% of human ground truth.

Do NOT loosen the three-row-and-most-of-the-page guard; a stray pipe in prose
would then mangle ordinary pages.

## 12. 701054 needs re-extraction; the other four do not (2026-07-31)

Measured, per §11. Re-segmenting the delivered volumes with the table fix:

| volume | table pages | delivered | re-segmented | delta |
|---|---|---|---|---|
| 176899 | 0 | 1,087 | 1,087 | 0 |
| 201991 | 2 | 2,085 | 2,090 | +5 |
| 29597 | 0 | 813 | 814 | +1 |
| 375062 | 1 | 1,137 | 1,140 | +3 |
| **701054** | **54 of 105** | **221** | **596** | **+375** |

701054 shipped missing ~63% of itself. ~$1.76 to extract the recovered entries
at the measured $0.0047/record, $2.80 to redo the volume; $4.15 headroom remains
under the $35 cap, so no new budget is needed. **PAID and Daniel's call** --
see `production/luna_v3/DM_701054_MISSING.md`.
