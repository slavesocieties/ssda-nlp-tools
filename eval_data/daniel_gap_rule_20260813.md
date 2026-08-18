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

---

## CORRECTION: the increment is 25%, not 44%

The corpus A/B says the rule is real and my headline number was **1.75x too high.**

| | pairs |
|---|---:|
| `blocked-context`, current default | 2,712,293 |
| `blocked-context`, gap rule on | 6,328,621 |
| **newly blocked by the rule** | **3,616,328 = 25.0% of candidates** |

**6,328,621 is the total blocked under the rule, not the increment.** 2,712,293 of
those pairs were *already* blocked by the existing 60-year window — every pair
more than 60 years apart is refused today. I counted pairs matching the rule's
criteria and reported it as pairs whose disposition changes.

That is §9 rule 4, verbatim: *"A BLOCK COUNT IS NOT AN IMPACT. The chronology
guard blocked 1,416 merges; 1,305 were already blocked by another rule."* The
handover names this trap with a worked example and I walked into it anyway,
inside a measurement built to price a change.

### What the rule actually does

| | |
|---|---:|
| pairs newly blocked | 3,616,328 (25.0% of candidates) |
| ...would have scored below review | 3,506,838 |
| ...would have been lifespan-vetoed anyway | 104,745 |
| ...would have reached review | **665** |
| ...would have AUTO-MERGED | **0** |
| identities | 33,180 → **33,180**, unchanged |
| auto-merges | 6,273 → **6,273**, unchanged |
| review pairs | 629,209 → **628,544** |

**The delivered answer does not move at all.** Same people, same merges, 665
fewer pairs put in front of a historian — and he graded a sample of exactly those
25/25 as noise.

Note the review-pair loss is **665**, not the 814 I predicted from scoring pairs
in isolation. The pipeline's cluster guards and blocking already removed 149 of
them. Score is not disposition, again.

### On runtime: no claim

The two runs clocked 966s and 952s, and **that comparison is worthless** — they
were not taken serially under controlled load, which HANDOVER §15 records as the
one measurement contention can corrupt. Counts are safe; timings are not. Anyone
wanting the efficiency figure must run both arms serially on an idle machine.

What can be said without a timing: **3.5 million fewer pairs reach the scorer**,
because `_shares_context` short-circuits before `score()` is called. Whether that
translates to wall-clock proportionally is unmeasured.

### The claim, corrected

**25% fewer pairs scored, no change to the delivered result, 665 fewer review
pairs, all of which the supervisor has confirmed are noise.**

Still worth doing. Just not 44%.
