# `W_VOLUMES_NEVER_COEXIST` is mostly double-counting — 2026-08-13

The widest-reaching non-name weight in the model: **−1.5, firing on 52.1% of
scored pairs**, and HANDOVER §4 lists it as *"still a prior; not separately
measurable here"* — chosen, never calibrated. Nothing that shapes half the corpus
should be invisible, so this looks at what it is actually doing.

## The overlap with the year term

Two registers whose date ranges never meet hold records that are, nearly by
construction, far apart in time — which `W_YEAR_FAR` already penalises.
Cross-tabulated over 197,213 scored pairs:

| coexist fires | year term | pairs | % of scored |
|---|---|---:|---:|
| no | positive | 80,243 | 40.7% |
| **yes** | **negative** | **63,816** | **32.4%** |
| yes | neutral | 23,395 | 11.9% |
| yes | positive | 8,770 | 4.5% |
| yes | one side undated | 6,770 | 3.4% |

Of the 102,751 pairs where the penalty fires:

- **62.1% also take a negative year term.** Both charge for the same underlying
  fact, −1.5 and −0.32 together, and the model is paying twice.
- **6.6% have an undated side.** This is the only place the weight adds a fact
  the year term cannot: 2.4% of mentions carry no year, and the volume's range
  still dates them. Here it genuinely earns its keep.
- **8.5% take a POSITIVE year term** while the penalty fires. The two signals
  disagree outright: the records are close in time, the volumes are said never
  to coexist.

## The hypothesis I had, and why it was wrong

Those 8,770 disagreements looked like a metadata defect. Comparing claimed volume
ranges against actual entry dates seemed to confirm it — **6 of 7 volumes hold
entries far outside their claimed range**:

```
176899  claimed 1887-1889   actual entries 1856-1917
201991  claimed 1839-1852   actual entries 1750-1899
29597   claimed 1770-1792   actual entries 1677-1885
```

That reads as damning: a −1.5 penalty on half the corpus, driven by ranges the
corpus contradicts.

**It is wrong.** Min and max are outlier statistics. The distribution says the
opposite:

| volume | claimed | p5 / p50 / p95 | inside claim |
|---|---|---|---:|
| 176899 | 1887-1889 | 1887 / 1888 / 1889 | **98.7%** |
| 201991 | 1839-1852 | 1839 / 1845 / 1859 | 87.8% |
| 29597 | 1770-1792 | 1771 / 1784 / 1792 | 97.0% |
| 701054 | 1865-1877 | 1865 / 1867 / 1870 | **100%** |
| 701179 | 1739-1751 | 1740 / 1744 / 1764 | 91.6% |

**87.8% to 100% of entries fall inside the claimed range.** The wide spans come
from a thin tail — `_year` is the entry's *earliest* event, so a burial recording
a birth decades earlier dates the entry to the birth. The metadata is accurate
and the same tail explains the 8.5% disagreement.

I nearly reported that the model's widest weight fires on false premises. Two
numbers (min, max) said yes; the distribution said no. **A range is not a
distribution**, and outlier-driven statistics are exactly what a chosen weight
should not be re-litigated on.

## What stands

The weight is **not** firing on bad data. It **is** largely redundant: on 62% of
its firings the year term is already penalising the same fact.

That is a calibration question, not a bug, and it belongs to whoever revisits the
conservatism dial. Two honest framings, and the choice between them is Daniel's:

- The redundancy is **deliberate conservatism**. Charging twice for "far apart in
  time, and in books that never overlap" is a stance, not an error, and Daniel
  has said he would rather miss a match than make a wrong one.
- Or it is **a prior doing more work than anyone intended**, since nobody chose
  −1.82 as the combined penalty; they chose −1.5 and −0.32 separately.

Worth knowing before anyone tunes either number in isolation.
