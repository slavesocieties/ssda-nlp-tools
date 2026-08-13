# The co-participant defect is live, not latent — measured 2026-08-13

`measure_latent_coparticipant.py`. Companion to
[associate_llr_null_20260813.md](associate_llr_null_20260813.md).

## What HANDOVER §14 says

> "this is not a live defect — it is a **latent** one. The moment the corpus
> contains two registers covering the same parish, co-participants will start
> appearing in different entries, and nothing will stop them merging."

The reasoning was that these 7 volumes come from 7 different institutions, so
the situation cannot arise yet. **That reasoning has a hole**: the defect does
not need two registers covering one parish. It needs **one event recorded in two
entries**, and the corpus already contains that.

## The free label nobody used

Duplicate entries *are* one event recorded twice. Across the two copies:

- **local id `i` vs `i`** — the same person. Should merge.
- **local id `i` vs `j`** — different people, co-participants in one event,
  in **different entries**, so the same-entry veto never fires.

That is precisely §14's scenario, available today at no cost.

## Result on the raw assembly

62 duplicate groups covering 153 entries (grouped on the **extracted people
payload** — names, local ids, relationships — *not* byte-identity; that is
deliberately a looser set than `dedupe_entries.py`'s 59 byte-identical records,
and the two counts must not be quoted as the same quantity).

| | pairs | scored | auto-merge |
|---|---:|---:|---:|
| same person (should merge) | 244 | 198 | **170 (85.9%)**, 99.5% at least review |
| different people (must not) | 529 | 208 | **50 (24.0%)** |

Of the 50 false merges, **27 have identical names and 23 differ**. The
identical-name half is the weak half: the premise that two local ids are two
people is the *extractor's* separation, and a namesake split in two would look
exactly like this. The 23 different-name pairs are the defensible core, and the
mechanism is visible in them:

```
+6.13  Paula Corrales      vs  María Corrales
+4.97  Juan de la Cruz     vs  María de la Cruz
+4.78  María Corrales      vs  Juan Corrales
+4.38  Antonio Rivas       vs  María Rivas
```

Siblings. Same surname, different given names, sharing parents — exactly the
co-participant signature §14 predicted, confirmed rather than assumed.

## It survives de-duplication — at the pair level

> **Read the corpus A/B at the end before quoting this section.** Everything
> here is measured at the *pair* level. The pipeline turns out to be protected
> anyway by a later guard, so "survives into production" below means "survives
> into the scored candidate set", not "produces a wrong identity". That
> distinction is §9 rule 3 and I did not have it right when I wrote this
> section.

`dedupe_entries.py` collapses **byte-identical records only**. Two entries whose
transcription text differs slightly but whose extracted people are identical are
not collapsed. So on `assembled_deduped` — **the default corpus since
2026-08-12**:

| | |
|---|---:|
| duplicate groups remaining | **28**, covering 60 entries |
| false auto-merges | **5** |
| of which genuinely different names | **3** |

```
+5.13  María de la Concepción    vs  José María de la Cruz
+4.12  María Dolores             vs  María de la Cruz
+3.08  Catarina de Vasconcellos  vs  Amaro de Freitas
```

**So this is a current defect in the shipped configuration, not a future risk.**
Small — 3 to 5 pairs — but the mechanism scales with the archive, and it needs no
same-parish register pair to fire.

## What the repair is

Not a re-weighting. The companion measurement shows the associate weight is
correct — even conservative — in the population the scorer meets. The repair is
to make the **same-entry veto a same-EVENT veto**: two entries whose extracted
people payloads are identical are one event, and pairs across them should be
vetoed exactly as pairs within one entry are.

The grouping is already written and cheap. It has a validation set (these 62
groups) and a directly measurable success criterion: the 50 false merges go to
zero while the 170 true merges survive, since same-id pairs across copies must
still merge.

## How wrong I got this first

The first run of this measurement reported **130** false merges. The honest
figure is **50**. Two bugs, both inflating, both mine:

1. Iterating `A × B` visited `(i,j)` and `(j,i)` — every pair counted twice.
2. A group of 3–5 copies produces several copy-pairs of the *same two people*,
   so counting per copy-pair counted one confusion up to four times.

Then a third of the remainder dissolved under scrutiny: 27 of 50 have identical
names, where the guaranteed-different premise is weakest.

§9 rule 4 says *pairs are not people, state the unit*. This is that rule costing
a 2.6× inflation, in a measurement built specifically to test someone else's
claim carefully. The script now counts one representative copy-pair per group,
unordered, and splits identical-name from different-name in its own output so
the weak half cannot be quoted as the strong half.

## Reproduce

```bash
python measure_latent_coparticipant.py --out <outside production/>
```

Refuses `assembled_deduped` by default: de-duplication removes the second copy,
so the measurement would report near-zero and read as "no problem". The 28
surviving groups above were measured by grouping that corpus directly.

---

## Corpus A/B: the veto changes nothing that matters

Measured after the fact, on `assembled_deduped` with keyed blocking, both arms
run serially. **This deflates the finding above and is the number to quote.**

| | control | `--same-event-veto` |
|---|---:|---:|
| identities | 33,179 | 33,179 |
| auto-merges | 6,274 | 6,274 |
| review pairs | 629,212 | **629,204** (−8) |
| merged identities | 1,118 | 1,118 |

The veto fires 36 times and moves 8 review pairs. Zero identities, zero
auto-merges.

**Why**, from the veto counts:

```
veto-cluster-same-entry   612,388  ->  612,374   (-14)
veto-same-event                 0  ->       36
```

`veto-cluster-same-entry` was already catching these **transitively**. Merging
P01 from copy A with P02 from copy B would place two people who share an entry
into one cluster, and that is refused regardless of what the pair scored.

So the honest statement is narrower than the section above implies:

- **The scorer is genuinely wrong on these pairs** — 24% of the scorable ones
  clear auto-merge on evidence alone. That part stands.
- **The pipeline was already protected.** §14's "the same-entry veto is doing
  the heavy lifting, and it is doing it well" survives this better than my
  finding did.

§9 rule 3 — *score is not disposition; guards run after scoring* — is what
separates those two sentences. Measuring at the pair level and reporting the
result as a production defect would have overstated it by everything except 8
review pairs.

### So does the flag earn its place?

Only on one argument, and it is not yet tested: §7d says the cluster guards are
**order-dependent**, evaluated against the cluster as built so far. If that holds
here, the protection currently relied on is path-dependent while a pairwise veto
is not — the same distinction that made the `name_similarity` symmetry fix worth
doing at a delta of 6 identities.

That is a claim about stability, so it is testable with this project's own
technique: run both arms under `--shuffle-seed` and see whether
`veto-cluster-same-entry` moves across orderings while `veto-same-event` stays
pinned at 36. **If the cluster guard is stable, the flag is redundant and should
be deleted rather than kept for tidiness.**

### The stability argument fails — shuffled A/B, seed 7

| run | identities | auto | review | `veto-cluster-same-entry` |
|---|---:|---:|---:|---:|
| control | 33,179 | 6,274 | 629,212 | 612,388 |
| control `--shuffle-seed 7` | 33,179 | 6,274 | 629,212 | 612,388 |
| same-event | 33,179 | 6,274 | 629,204 | 612,374 |
| same-event `--shuffle-seed 7` | 33,179 | 6,274 | 629,204 | 612,374 |

**Identical in every column.** The cluster guard is not behaving
order-dependently here, so the last argument for the flag — that it replaces a
path-dependent protection with a deterministic one — is unsupported.

**On current evidence `--same-event-veto` is redundant and should be deleted
rather than kept for tidiness**, which is the criterion set before the run. Two
things stop that being the final word, and both are stated rather than used as
cover:

1. **One seed is weak evidence of stability.** A single shuffle agreeing proves
   less than the claim it was asked to settle. Seeds 1 and 2 would cost ~26
   minutes and would make this conclusive either way.
2. **It is the only implementation of a mechanism §14 predicts will be needed**
   — though not in this form, since it relies on a shared local-id scheme the
   two-registers case will not have.

### A contradiction with §7d worth chasing

§7d states identity counts vary by **±6 across shuffles** and that the cluster
guards are order-dependent. Measured here: **zero variation**, on both arms.

Either §7d is stale — plausibly fixed by the `name_similarity` symmetry work or
the blocking rewrite, both of which landed after it was written — or the spread
needs more than one seed to surface. That is a live discrepancy in the handover,
not a settled correction, and it is the cheaper of the two open questions above.

### The identical-name half is stronger than I allowed

I discounted the 27 identical-name false merges on the grounds that "two local
ids means two people" is the *extractor's* separation and a namesake split in two
would look the same. Measured rather than assumed, across every within-entry pair
sharing a normalised name in the raw assembly:

| | count | reading |
|---|---:|---|
| genuinely different relationships | **641 (83%)** | really two namesakes |
| one side has no relationships | 68 (9%) | ambiguous |
| identical relationships on both | 60 (8%) | indistinguishable; likely one person split |
| **total same-name within-entry pairs** | **769** | |

So the premise holds for about **83%** of same-name pairs. Applying that rate,
roughly 5 of the 27 are plausibly extraction duplicates and ~22 are real, which
puts the defensible count nearer **45 of 50** than the 23 I retreated to.

State it as a range: **at least 23** (different names, premise unimpeachable),
**probably ~45**. My caution was right in direction and overdone in size — the
same error as the 130, in the opposite direction.

### A separate defect this turned up

**60 within-entry pairs share a name AND hold identical relationships**, plus 68
where one side holds none. HANDOVER §3 records exactly **one** duplicate person
id inside an entry (`701157-0214-01`). If even half of these 60 are one person
emitted twice, the extractor's within-entry duplication rate is ~60×
what is documented, and every one of them inflates a person count and seeds a
phantom identity.

Not asserted: two namesakes sharing one relationship is possible (siblings both
"child of X", a dead child's name reused), so 60 is an **upper bound**. It is
cheap to settle — the 60 are enumerable and readable — and nobody has looked.
