# Daniel's 10% covered 0.066% — the grading page hid what a row was worth

Daniel returned `blocked_labels.json` on 2026-08-13: **20 of 200 rows, every one
graded 0**, with the note that he suspects they are all good discards. They are.
The problem is not his answer, it is what the answer can be extended to.

## The numbers

He graded rows **0–19**, the first twenty, which is the only sensible way to
approach an unordered list.

| | rows | pairs represented | share of the 3,013,215 pool |
|---|---:|---:|---:|
| what was graded (rows 0–19) | 20 | **1,995** | **0.066%** |
| the 20 heaviest rows | 20 | **1,692,576** | **56.2%** |
| enough to cover half the pool | **17** | 1,506,608 | 50% |

**Grading the heaviest twenty would have been worth 848× more.** Same effort,
same number of judgements.

## Why it happened

The sample is Horvitz-Thompson weighted — that is the whole point, and it is why
200 rows can stand for three million pairs. Row weights span **1 to 134,583**.

`build_blocked_labels.py` emitted rows in `sorted(size)` order, which sorts
strata by **name**, scattering weight arbitrarily through the page. And the page
never showed the weight: the only occurrence of the word "weight" in
`blocked_pairs.html` is a CSS `font-weight`. So the grader had no way, from
inside the task, to tell a row worth 134,583 pairs from one worth 3.

The instrument silently answered a much narrower question than the one being
asked — the same shape as the inert grandparent capacity, the mtime that could
not distinguish a touch from a rewrite, and `git fetch <sha>` failing whether or
not an object exists.

## What the estimate actually says

```
point estimate      : 0 hidden merges
UNREPRESENTED       : 3,011,220 pairs (99.9%) in strata with NO graded row
```

Zero, over 0.066% of the pool. `estimate_blocked_recall.py` reports the
unrepresented mass rather than summing it in as zero, which is the only reason
this reads correctly instead of as **"recall is perfect, no merges are being
lost."** That sentence was one naive `sum(weight × grade)` away.

## Fixed for future draws

- Rows are now emitted **heaviest stratum first**, so a partial grading — the
  normal case for a busy supervisor — buys the most population per judgement.
- The page now tells the grader the rows are ordered by what they stand for and
  that the first 17 cover half the pool.

**The delivered page was not regenerated.** It is locked, Daniel's grades are
keyed to its row positions, and reordering it would invalidate the 20 answers we
have. The fix applies to the next draw.

## The ask that is worth making

Daniel's 20 zeros are real evidence and worth having. To turn them into a recall
number rather than an anecdote, the cheapest possible follow-up is:

> **grade rows 79–103 in the same page** — 20 more judgements, and the covered
> share of the discarded pool goes from 0.066% to about 56%.

If they come back zero as well, that is a genuine result: the pre-filter is
discarding almost nothing real, measured over half the pool rather than a
thousandth of it.
