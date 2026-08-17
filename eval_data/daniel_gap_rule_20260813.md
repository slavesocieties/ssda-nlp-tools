# Daniel's gap rule: 43.7% of candidates for 814 review pairs — 2026-08-13

> *"they could all be ruled out by dates, and don't really require any contextual
> knowledge to inspect. The only case in which a pairing >50 years apart, or
> frankly even half of that, should even be 'looked at twice' (i.e. passed to
> more complex algorithmic processes) is if there is potential overlap in social
> networks or complete alignment in name/phenotype/ethnonym etc."*
> — Daniel, 2026-08-13

A ruling about **blocking**, not scoring. `_shares_context` keeps a pair on ANY
of: same register, a person named in both, or dated within 60 years. The third
clause is permissive — a pair survives on the year window alone with no register
in common and nobody shared.

## What is currently kept on the year window alone

| | pairs | share of candidates |
|---|---:|---:|
| same register (his rule does not touch these) | 5,664,053 | 39.1% |
| **cross-register, year window only, >50y** | **3,455,949** | 23.9% |
| **cross-register, year window only, 26–50y** | **2,875,668** | 19.9% |
| cross-register, year window only, ≤25y | 1,923,228 | 13.3% |
| cross-register, undated | 545,711 | 3.8% |
| cross-register, shares a person | 7,404 | 0.05% |

**57% of all candidate pairs survive on the year window alone.**

Applying his rule at **>25 years**, with his exceptions — exact name **and** every
stated attribute agreeing — rescuing 2,996 pairs:

**6,328,621 candidate pairs would be dropped. 43.7% of the entire set.**

## What it would cost

Every one of those 6,328,621 scored, not sampled:

| | pairs |
|---|---:|
| already vetoed anyway | 56,307 |
| below review — no loss | 6,271,500 |
| **would lose a review pair** | **814** |
| **would lose an auto-merge** | **0** |

**Zero merges lost.** The cost is 814 review pairs, 0.013% of what is dropped.

And those 814 are exactly what he said should not be surfaced: cross-register,
sharing nobody, more than 25 years apart, without complete name and attribute
alignment. The cost is not a regrettable side effect of the rule — it *is* the
rule, doing what he asked.

## Why this is well-supported rather than a guess

Three independent lines agree:

1. **His judgement**, twice over: 73 graded pairs, all zero, and *"nothing close
   in this set"*.
2. **The recall measurement**: zero true merges hidden across 100% of the
   already-blocked pool. The pairs this rule would newly drop are the same kind —
   cross-register, sharing nobody, far apart in time.
3. **The scorer already refuses them**: 99.1% score below review today, so the
   model and the historian agree about this population.

## The scale argument

Candidate pairs at the full collection go from ~1.33e10 to roughly **7.5e9**, and
the drop is in the class that grows worst — cross-register pairs with no shared
context, which multiply as volumes are added, unlike same-register pairs that
stay bounded within one book.

## What NOT to do with this

**Do not implement it silently.** It is a blocking change, and this project has
learned that the hard way: keyed blocking's losslessness lapsed once, silently,
and was restored only because `verify_blocking.py` exists. Any implementation
needs the same treatment — `verify_blocking` re-run, a full corpus A/B, and the
814 lost review pairs read rather than counted.

**And it is a 7-volume result.** The rescue exceptions save 2,996 pairs here; at
3,900 volumes, with far more same-name people and more registers per parish, both
the drop and the rescue change size. HANDOVER §13(b) makes the same point about
`max_block` and it applies with equal force here.

## Reproduce

```bash
python measure_gap_rule.py          # the benefit side
```

The cost side scores all 6.3M dropped pairs and takes ~10 minutes; the script is
in the session scratchpad rather than committed, since it is a one-off.
