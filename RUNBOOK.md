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

## 13. 701054 re-extraction — staged and validated, NOT sent (2026-08-01)

Daniel approved this. Segmentation is redone and the batch is staged; only the
paid submission remains.

```
production/corpus_v6/701054.segmented.json   596 entries (was 212)
production/batches_v6/701054.batches.jsonl   60 requests, ~$1.69 Batch API
```

Validated before staging was called done: 596/596 segmentation ids present in
the batches, all unique, custom_ids unique and mapping to volume 701054, and the
instruction reads **"Process ALL 10 Portuguese burial entries"** rather than the
"Spanish baptism" that every previous batch said.

**That prompt defect is general, not a 701054 typo.** `build_messages` defaults
to `record_type="baptism", language="Spanish"` and no caller ever passed them, so
every volume so far was told it was Spanish baptisms. It did no measured damage
-- 701054 still extracted 187 burials, 29597 723 marriages -- because the system
prompt says explicitly not to translate and the model read the text. Pass
`--language` / `--record-type` from now on; the flags already exist.

> ### ⚠️ The trap, the same one as §2: 210 of the 596 entry ids already exist
> The delivered run covered 210 of these entries under the SAME ids. Assembling
> the new output into `production/luna_v3/` would collide with them. Use a fresh
> `--outdir`, a distinct `--run-id`, and point `--ledger-path` at the ONE
> cumulative ledger so the $35 cap stays global.
>
> The batch custom_ids (`701054-b0000`…) do NOT collide with anything in the
> ledger -- checked, 0 of 60 -- because the delivered run used `v3-` prefixes.
> The collision is at ENTRY id level, in assembly, not at submission.

```bash
python run_luna_production.py production/batches_v6/701054.batches.jsonl --outdir production/luna_v6 --ledger-path production/luna_live/spend_ledger.json --run-id v6 --cap-usd 35.00 --take 60
```

Run it once WITHOUT `--confirm` first. Headroom is $4.15 against the $35 cap
before the two outstanding jobs settle, and this needs ~$1.69, so settle
701157/701179 first or the reservation may not fit.

## 14. Verification state, and the one stage that has none (2026-08-02)

| stage | verified against | result |
|---|---|---|
| transcription | human gold, 3 vols | substitution 6.31%, median similarity 0.891 |
| transcription integrity | measured | 1,281 failed pages, 100% precision |
| segmentation | human gold, 3 vols | **98.2%** of human entry count |
| **extraction** | **nothing** | no usable human gold exists |
| normalization / vocab | self-consistency | ethnicity 100%, age 100% |
| merge | 4 gold negatives + logic | all held; labels pending |
| social graph | logical invariants | 21 impossible identities -> 0 |
| corpus QA | triaged | 531 real defects of 5,226 (10.2%) |

**Extraction is the gap and it cannot be closed from the repo.**
`slavesocieties/openai` has ~86 entries with structured `data`, but ~65 sit in
`testing/` under names like `*_driver_output` -- model output, not gold.
Measuring our extraction against another model's measures agreement, not
accuracy. ~21 usable records for the stage that produces every person and event.

```bash
python audit_corpus.py --stage production/repair_YYYYMMDD   # triaged defects
python validate_graph.py                                    # graph invariants
python run_manual_gold.py                                   # transcription vs human
python run_seg_gold.py                                      # segmentation vs human
python run_gold_merge.py                                    # merge vs 59 hand labels
```

**Trap: two identity-resolution paths.** `run_merge.py` -> `disambiguate.py`;
`run_pipeline.py` -> `link.py` -> `resolve.py` -> `disambiguate.py`. They
converge, so a disambiguate change DOES reach the delivered graph -- but only
after `assemble_corpus.py` runs WITHOUT `--skip-pipeline`, which takes 30+ min.
An unchanged `network.json` usually means the rebuild has not finished.

## 15. Graph after the contradiction guard — what it fixed, what it did not

Rebuilt 2026-08-02 (`assemble_corpus.py` without `--skip-pipeline`, ~10 min).

|  | before | after |
|---|---|---|
| identities carrying mutually exclusive attributes | 21 | **0** |
| max node degree | 208 | **112** |
| people with degree > 50 | 9 | **5** |
| distinct people | 22,702 | 22,740 |

"María de la Cruz" split into separate people (112 and 58) instead of one
82-mention node holding both free and enslaved, infant and adult.

**STILL OPEN, and it is a DIFFERENT mechanism: 64 contradictory role pairs and
26 ancestry cycles.** Nothing to do with attributes. Example:

    CORPUS-4204 'Ramona Bernal'  --parent--> CORPUS-4208 'Rosalía Bernal'   [176899-0017-B-02]
    CORPUS-4204 'Ramona Bernal'  --child-->  CORPUS-4208 'Rosalía Bernal'   [176899-0195-B-03]

Both women are *parda*, both from Trinidad, two mentions each, no attribute
conflict at all -- so the attribute guard cannot see it. Folios 17 and 195 are
far apart, so the likeliest reading is TWO different mother/daughter pairs
sharing the surname Bernal, collapsed by name similarity.

**Done 2026-08-03, and it is a PARTIAL fix -- read the numbers, not the
headline.** `_would_close_ancestry_cycle` refuses a merge when a descent path
already runs between the two clusters. Measured on the delivered graph:

| invariant | before | after |
|---|---|---|
| parent AND child on the same pair | 16 | **0** |
| ancestry cycles of length 2 | 12 | **0** |
| ancestry cycles of length 3 | 12 | 9 |
| ancestry cycles of length 4 | 4 | 6 |
| ancestry cycles of length 5 | 0 | 1 |
| **total ancestry cycles** | **28** | **16** |
| contradictory role pairs (graph) | 65 | 27 |

So it fully closes the 2-cycles it was designed for and cuts the total 43%, but
LONGER CYCLES PERSIST AND SOME GREW. That is order dependence, not a bug to wave
away: the guard refuses a merge that closes a loop VISIBLE AT THAT MOMENT, and a
later merge can complete a longer one through clusters joined afterwards.
Blocking a short loop sometimes reroutes the same over-merge into a 4- or 5-hop
one.

I first reported this as "28 -> 0". That was wrong: the check counted only
2-cycles and double-counted ordered pairs. Any claim about this guard must come
from `validate_graph.py` on the rebuilt graph, not from a bespoke count.

`python validate_graph.py` is the check; it exits non-zero while any invariant
fails.

## 16. 701157 + 701179 assembled from already-paid output (2026-08-03)

Found by `verify_claims.py`, not by looking: the spend ledger had moved from
$24.53 to $29.15 confirmed, and chasing that turned up ~$3.57 of extraction
sitting on disk unassembled in `production/new_volumes/live/`.

```bash
python assemble_corpus.py --live production/luna_v3 \
  --accepted-dir production/luna_v3 --accepted-dir production/new_volumes/live \
  --corpus production/corpus --corpus production/new_volumes
```

**Corpus 5,226 -> 6,794 records, 5 -> 7 volumes.** 701157 adds 872 (848
marriages), 701179 adds 696 (744 baptisms). Zero integrity failures in either.

> ### ⚠️ Assemble against the OLD segmentation, not corpus_v6
> The paid extraction ran against `production/new_volumes/*.segmented.json`.
> Entry-id overlap is **100% (1,570/1,570)** against that and only 98.5% against
> the table-fixed `corpus_v6`. Using the newer segmentation would silently drop
> 23 paid records. Re-segmenting these two volumes is worth doing eventually --
> it is +2 entries each -- but it needs its own small re-extraction.

**The generalisation gap is now measured, not hypothesised.** These are the
first substantially Portuguese volumes and ethnicity conformance falls from
100% to 61% (701157) and 45% (701179). The strays are real Brazilian ethnonyms
(Guiné, Nação, Benguella, gentio da Costa/Guiné/Angola), Portuguese phenotype
spellings (crioulo/crioula), and 966 Portuguese titles (Reverendo 441,
Coadjutor 325). Needs Daniel's ruling exactly as the 71 ethnicity terms did --
see `production/luna_v3/DM_NEW_VOLUMES_VOCAB.md` and
`new_volume_vocab_gaps.json`.

`sibling` (20) and `cousin` (2) come out of the extractor as relationship types
the schema does not know, which is why 4 of the 6 missing graph inverses are
sibling edges.

## 17. Graph at 7 volumes, and the extraction/merge split

32,628 people, 53,800 edges. Rates per 10k edges, 5-vol -> 7-vol:

| invariant | 5-vol | 7-vol | per 10k edges |
|---|---|---|---|
| self loops / dangling endpoints | 0 | **0** | |
| contradictory roles | 4 | 8 | 1.05 -> 1.49 |
| ancestry cycles | 12 | 15 | 3.14 -> **2.79** |
| missing inverse | 1 | 6 | 0.26 -> 1.12 |

`audit_corpus.py` now separates the two causes, which matters because they need
different fixes: **same-entry role contradictions are EXTRACTION defects and
there is exactly 1 in 6,794 records** (701179-0148-01, mutual parenthood). Every
other role contradiction in the graph is a merge artifact. Do not conflate them.

## 18. Why re-extraction keeps appearing on the list (2026-08-05)

The question that prompted this section: *why are we doing this again and again?*
Checked rather than assumed, and the answer is that two unlike things had been
filed under one heading.

**We are NOT re-paying for the same records.** Zero records carrying a repair
provenance marker are still defective. There is no loop, and nothing has been
extracted twice.

**"363 records" was wrong, and I had been repeating it.** 363 is the ISSUE count.
It resolves to **244 distinct records**, 3.6% of the corpus, because one record
commonly carries several dangling relationships (1.5 issues each). The free-repair
figure has the same shape: 282 issues, **180 distinct records**. Quote records
when the unit of work is a record.

**The count grows because the CORPUS grows.** The issue rate has been flat-to-
improving: 301 issues on 5,226 records (5.8%) at 5 volumes, 363 on 6,794 (5.3%)
at 7. Every new volume brings its own share of records the extractor misread.
That is a property of a ~96% accurate extractor, not a regression, and it will
keep happening for as long as volumes keep arriving.

**It is heavily concentrated, which is worth knowing BEFORE paying:**

    volume    records   need re-x    rate
    375062      1,132          88    7.8%
    29597         780          57    7.3%
    176899      1,085          34    3.1%
    701179        696          23    3.3%
    701157        872          13    1.5%
    201991      2,019          27    1.3%
    701054        210           2    1.0%
    ALL         6,794         244    3.6%

29597 and 375062 are 145 of the 244 between them, from 28% of the corpus. Two
volumes are six times worse than 201991. Find out why before buying a fix.

**701054 IS NOT IN THIS CATEGORY AT ALL.** It is not a re-extraction. Those 375
records were never extracted once, because OUR segmenter skipped 54 of its 105
pages: the registers are physically two-column, Gemini transcribes that as a
markdown table, and any page full of pipe characters was classified as an index
page and dropped (see §11, §12). We delivered 221 of 596. The staged 61 requests
finish a volume we under-delivered through our own bug, which is now fixed. It is
a one-time completion and it will not recur.

So the standing list should read:

    244 records   extraction misread the text          recurring, ~3.6%/volume
    375 records   we never sent them (segmenter bug)   one-time, bug fixed
    180 records   repairable for free                  no spend at all

## 19. Today's changes (2026-08-05)

Commands added; all offline, $0, no key.

```bash
# A/B two merge runs, and REFUSE the comparison if it is not valid
python compare_merge_runs.py v8control2 v8lifespan

# re-score Daniel's labels against the delivered run (control runs first)
python verify_label_scores.py

# per-rule cost of the merge bar, corpus-wide
python analyze_surname_tradeoff.py --tag v9tradeoff2

# the relationship review queue (the weakest extracted field)
python run_relationship_review.py --limit 500

# is the training sample actually a fair stratified draw?
python validate_training_sample.py
```

**Rebuild traps, both of which cost time today.**
`assemble_corpus.py` defaults to `--live production/luna_live` -- the OLD 5-volume,
5,226-record corpus. For v3 use the invocation at §16, or, to regenerate only the
graph, `run_pipeline.py production/luna_v3/assembled/*.materialized.json --outdir
<fresh dir>`. And `corpus_final_pipeline/` PREDATES the current merge guards
(32,628 nodes vs 32,943), so any before/after diff against it is confounded --
verify with `validate_graph.py` on the new graph, never by delta.

**Results.** Chronology guard: 1,416 blocks, but 1,305 were already refused by the
surname tiers; 111 pairs and **33 people** are attributable to it alone.
Merge vs Daniel's certain labels: **21/24 (88%)**. Graph `missing_inverse` 6 -> 2
after completing SELF-INVERSE types only (sibling/spouse/witness -- parent/child
would mean inferring a direction). Training sample sound: weights reconstruct
7,305,667 pairs to 0.000% error. Security audit clean: no credential material in
the tree or in 140 commits.

**Two counting bugs fixed in the tools that report counts.**
`validate_graph.contradictory_roles` was keyed on the ORDERED pair and read 8 for
5 distinct pairs. The tidier fix -- unioning both directions before testing --
takes it to **10,050**, because A->B "parent" plus B->A "child" is every real
parent. Detect per direction, report once.
`_merge_attributes` deduped on the raw string while the scorer normalises through
`_val`, so "Cleric" vs "cleric" was a reported conflict on 7 of the 12 largest
clusters. Decisions were never affected; the report diverged from the rules.

## 20. Why 375062 and 29597 are worse (2026-08-05)

Two different causes. One is explained, one is not, and I am labelling them that
way rather than giving both a story.

**The mechanism, corpus-wide.** In 95-100% of dangling relationships the
referenced id is BEYOND the number of people the extractor itself listed, with a
median gap of exactly 1: it writes N people, then refers to P(N+1). This is a
truncation pattern, not random invention.

**Exposure is relationship density, and it is monotonic:**

    rels/entry   entries   w/ dangling   rate
         0-2       1,840          2      0.1%
         3-5         564          2      0.4%
         6-8       1,596         12      0.8%
        9-11       1,235         31      2.5%
       12-14         745         21      2.8%
        15+          814         53      6.5%

**29597 is EXPLAINED by that.** It has the densest entries in the corpus, mean
13.7 relationships against 201991's 3.4. Against the pooled rate for its own
density it runs 1.40x -- somewhat high, not anomalous. Its defect rate is what a
volume of that shape should produce.

**375062 is NOT explained.** It runs **2.23x** what its density predicts, and it
also has the corpus's worst `no_people_real` rate (36.2 per 1,000, versus 12.4
for 201991) -- entries where the text plainly contains a sacrament and the
extractor returned nobody at all.

Reading those entries, the text looks damaged in a specific way ("mil ochocien
D. Miguel Llopiz", "esta Yglesia parroge. quial de Ingreso"), which suggests
margin text interleaved into the body. **I could not confirm it.** A mid-word-break
density proxy across all seven volumes does NOT track the defect rate -- 201991
scores highest on it (0.83) and has the LOWEST defect rate (1.3%) -- so the proxy
measures ordinary hyphenation, not corruption. Treat the interleaving story as an
unverified hypothesis.

**The consequence for spending.** For 29597, re-extraction should help: the
entries are legitimately dense and the extractor drops the tail. For 375062, the
cause is unknown, and re-running the same extractor over the same text may well
reproduce the same result. **Diagnose 375062 before buying a fix for it.** It is
88 of the 244 records.

## 21. The gold volumes and the delivered volumes are DISJOINT (2026-08-05)

    gold (accuracy measured):  15834, 1795, 419324
    delivered corpus:          176899, 201991, 29597, 375062, 701054,
                               701157, 701179
    overlap:                   NONE

So "transcription substitution 6.31%, median page similarity 0.891" and
"segmentation 98.2%" are measured on three volumes that are **not in the corpus
we ship**. They are evidence about the pipeline, not about the delivered data,
and every previous statement of them (including in PROJECT_STATUS §6) omitted
that. 375062 in particular cannot be checked against gold at all, which is
exactly why §20 above ends in "unknown" rather than a diagnosis.

This does not make the numbers wrong. It bounds what they cover.

## 22. 375062 diagnosed — the failure is ABOVE extraction (2026-08-05)

§20 left this volume as "worse than density predicts, cause unknown". Here is the
cause, from a within-volume comparison (which controls for everything that is a
property of the volume itself).

**It is not localised.** Defect rate by folio block runs 3.9%-12.5% across the
whole volume, with no hotspot. Not physical damage to a page range.

**All 41 `no_people_real` entries returned `{"people": [], "events": []}`** — a
well-formed, completely empty answer. Not a parse failure and not a missing
response. The extractor answered "nothing here" for baptism records. They split
cleanly in two:

**(a) 18 of 41 — the input was already broken.** The `normalized` field is empty
or is model COMMENTARY instead of a normalisation:

    375062-0006-A-03  faithful 1881 -> normalized 194
      "El texto está gravemente duplicado y truncado; contiene referencias
       incompatibles a varios registros ... No es posible establ[ecer]"
    375062-0013-A-03  faithful 1800 -> normalized 0
    375062-0031-B-04  faithful  760 -> normalized 0

The faithful text for these is itself damaged ("mil ochocien D. Miguel Llopiz",
"esta Yglesia parroge. quial de Ingreso"), and the normaliser is telling us so in
Spanish. 44% of the no-people set is truncated this way, against **1%** of the
volume's healthy entries.

**(b) 23 of 41 — the text is fine and extraction returned nothing anyway.**

    375062-0019-B-03  "En diez y ocho de noviembre de mil ochocientos setenta y
                       siete años, yo, presbítero Don Miguel Llopiz ... bauticé
                       solemnemente y ..."   -> {"people": [], "events": []}

Clean, well-formed, priest named, sacrament stated. Nothing about the input
explains the empty answer.

**375062 is enriched 3x on every normalisation failure.** It is 17% of corpus
records and:

    empty normalized   54% of the corpus total
    short normalized   51%
    model commentary   56%

**WHAT THIS MEANS FOR THE 88 RECORDS.** They are not one purchase.

    ~18  input is broken           RE-EXTRACTION CANNOT FIX THIS. The extractor
                                   would be re-reading the same empty or
                                   commentary-filled normalisation. Needs
                                   re-normalisation, and for the genuinely
                                   damaged ones, re-transcription.
    ~23  empty answer on good text A retry is reasonable; the model returned
                                   nothing on input that plainly supports an
                                   extraction.
    ~38  dangling relationships    The corpus-wide density pattern (§20).
    ~9   malformed events

So the honest recommendation for §18 item 8: **do not buy the whole 88 as one
line item.** Roughly a fifth of it is money spent re-reading broken input.

**A caution on the commentary count.** How many records carry model commentary in
`normalized` depends entirely on the regex: a broad pattern finds 9 corpus-wide,
a narrow one finds 1. Only 2 records are in `withdrawn_records.json`. Before
acting on that class, agree the pattern first -- the number is an artifact of the
detector, not a property of the corpus.
