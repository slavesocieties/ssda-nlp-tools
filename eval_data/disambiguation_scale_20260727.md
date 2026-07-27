# Disambiguation at corpus scale — measured, and a correction

Measured 2026-07-27 on the delivered corpus (5,228 records, 27,631 person
mentions). Contextual blocking is implemented and worth keeping, but it does
**not** solve the review-queue problem, and the estimate given to Daniel was
wrong by two orders of magnitude.

## What was claimed vs what is true

I told Daniel that filtering before scoring would take the queue "from ~796,000
pairs to low thousands of consequential decisions." Measured:

| configuration | review pairs | identities | comparisons skipped |
|---|---:|---:|---:|
| no blocking (baseline) | 795,713 | 16,230 | — |
| **context blocking (shipped)** | **612,495** | 16,800 | 3,322,325 |
| context blocking, fail-closed (rejected) | 594,193 | 16,881 | 3,626,925 |

That is a **23% reduction, not a 99% one**. 612,495 pairs is as unreviewable as
796,000. The "low thousands" figure conflated two different things: the number
of *consequential* merges with the number of *pairs presented*.

## Why blocking underperforms

The gate passes any pair sharing a register, and the corpus is five large
registers. Same-register comparisons therefore dominate and survive the filter
almost entirely. What blocking removes is mostly *cross*-register comparison —
the rarest and most valuable kind. It is still worth having (3.3M pointless
comparisons avoided, and the run is bounded), but it attacks the wrong term.

## The fail-open decision, and what it cost

The first implementation gated on metadata being *present*, so an undated entry
was excluded from every comparison and its genuine merges disappeared silently.
Eight existing tests caught it. The shipped version fails open: a pair is only
skipped on positive evidence it cannot be one person (different register, no
person named in both entries, and dated more than `year_window` apart).

Both versions still raise the identity count (16,230 → 16,800). Those ~570
suppressed merges were pairs failing all three gates simultaneously — different
register, no shared person, more than 60 years apart. Those are *probably*
false merges being correctly prevented rather than good merges being lost, but
that has not been verified record by record and should not be asserted.

## What would actually make review tractable

Not filtering. Reframing:

- **Per-person review, not per-pair.** 612,495 pairs is 16,800 person screens,
  of which only ~2,000 identities have any candidate at all. That is the
  arithmetic change; everything else is a rounding error against it.
- Pair review is also intrinsically repetitive: one identity with 40 candidates
  appears as 40 separate decisions, most of which a human answers once.

This also fits where Daniel said he wants to end up — a model that scores the
probability two mentions are the same, trained on manual decisions, with a
front-end that surfaces the uncertainty. Per-person screens are the natural
collection surface for that training data; per-pair queues are not.

## Correction owed

Daniel was told "low thousands". He should be told the measured number, that
blocking is a 23% improvement rather than a solution, and that per-person review
is the option that changes the order of magnitude.
