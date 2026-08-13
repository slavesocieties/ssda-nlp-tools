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

## The part that matters: it survives into production

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
