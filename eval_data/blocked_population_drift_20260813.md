# The sample Daniel is grading describes a corpus that changed under it

Checked because the lock cannot catch this. `delivered_labels.lock.json`
guarantees the **file** has not moved. It says nothing about whether the **world
the file describes** has moved — and it had.

`blocked_pairs.html` was drawn on 2026-08-10 from `production/luna_v3/assembled`,
the raw assembly, and its 200 Horvitz-Thompson weights sum to **3,013,215**, the
population of pairs that survive every guard and are never scored. On 2026-08-12
`assembled_deduped` became the default corpus. De-duplication removes 244
mentions, which changes which pairs get blocked — so the denominator the estimate
divides by may no longer be the denominator that ships.

## Measured

`analyze_blocked_pairs.py --assembled production/luna_v3/assembled_deduped`

| | raw (what the sample stands for) | deduped (the default) | drift |
|---|---:|---:|---:|
| **survives every other guard** | **3,013,215** | **2,952,727** | **−2.0%** |
| also lifespan-impossible | 4,298,601 | 4,239,812 | −1.4% |
| clergy | 1,232,129 | 1,213,577 | −1.5% |
| also both-sacrament-principals | 9,644 | 9,578 | −0.7% |

## What follows

**The sample stays usable and Daniel should not regrade.** The drift is 2.0%,
which is an order of magnitude smaller than the estimate's own uncertainty — on
200 rows with weights this uneven, the interval is hundreds of thousands of pairs
wide. Redrawing would discard grading he is doing right now to chase a bias far
below the noise.

**But the number must be reported for what it is:** an estimate of the merges
hidden in the **pre-de-duplication** blocked set. `estimate_blocked_recall.py`
now prints that caveat with every result, so it travels with the figure instead
of living in a document nobody re-reads.

## The general point

A lock over an artifact is not a lock over the configuration that produced it.
Both can invalidate a returned grade, and only one of them is checkable by hash.
When a delivered sample outlives a default change, measure the drift before
assuming either that it is fine or that it is ruined.

---

## Re-measured after the sided corpus was promoted

The default moved to `assembled_deduped_sided` on 2026-08-13, so the drift figure
above described a corpus that was no longer the default. Re-run.

**Predicted first, from the code.** `_shares_context` decides what gets blocked,
and its associate test is:

```python
{n for _, n in ac} & {n for _, n in bc}
```

It compares associate **names** and throws the relationship type away. Relabelling
a grandparent as maternal or paternal changes the type and not the name, so the
prediction was that the blocked population would be **identical**, not merely
close.

**Measured. Identical in every row:**

| | unsided | **sided (the default)** |
|---|---:|---:|
| survives every other guard | 2,952,727 | **2,952,727** |
| also lifespan-impossible | 4,239,812 | 4,239,812 |
| clergy | 1,213,577 | 1,213,577 |
| also both-sacrament-principals | 9,578 | 9,578 |

So the 2.0% drift against the sample's 3,013,215 is unchanged and still comes
entirely from de-duplication, not from the sided promotion. The estimator's note
now says so.

Worth doing as a prediction rather than just a re-run: a re-measurement that
merely agrees teaches nothing, while a prediction that survives measurement
confirms the mechanism as well as the number. Had it come back different, the
reading of `_shares_context` would have been wrong and that would have mattered
well beyond this figure.
