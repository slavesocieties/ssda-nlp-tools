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

## Per-person review — measured, not projected

Implemented in `ssda_nlp_tools/person_review.py` and measured on the same
corpus. Grouping the identical pair set by **resolved identity**:

| | count |
|---|---:|
| pairwise decisions | 612,495 |
| **person screens needing review** | **13,967** |
| identities with no candidate at all | 2,833 (of 16,800) |
| candidates per screen (mean) | 12.9 |

Screens by candidate count: 1 → 1,745 · 2–5 → 4,293 · 6–20 → 5,487 · 21+ → 2,436.

That is a **44x reduction in decision points**, and it is real: a person seen in
eight entries is one screen rather than eight, and a candidate reachable by
several paths is listed once. It is *not* the "low thousands" claimed earlier —
13,967 screens is substantial work, and the 2,436 screens carrying 21+
candidates are the genuinely hard tail.

Grouping by mention instead of identity gives 24,206 screens averaging 50.6
candidates, i.e. worse than the identity count it should produce. That was the
first implementation and it was wrong.

## Three wrong numbers, and why

This section is here because the same estimate was given wrongly three times.

1. "Filtering takes it to low thousands" — actually 612,495 (23% cut).
   Cause: predicted rather than measured.
2. "Per-person is ~2,000 screens" — actually 13,967. Cause: grouped by MENTION,
   not by resolved identity.
3. "1 screen, 16,799 need no review" — pure artifact. Cause:
   `person_index.json` keys identities as `person_id`; the code looked for
   `global_id`/`id`, so every lookup returned `None`, the corpus collapsed onto
   one screen, and every candidate was dropped as a self-match.

Only the third was caught by its own output being self-contradictory (1 screen,
0 candidates, yet counted in the "2–5" bucket). `mention_to_identity` now raises
rather than building a map of `None`s, and tests pin all three failures.

## What to tell Daniel

- Filtering before scoring is implemented and worth keeping, but it is a **23%
  reduction, not a solution** (795,713 → 612,495).
- Per-person review is the real change: **612,495 pair decisions → 13,967 person
  screens**, average 12.9 candidates each, 2,833 identities needing nothing.
- It is still meaningful work, concentrated in the 2,436 screens with 21+
  candidates. Reviewing highest-probability screens first is how that tail gets
  managed — which is the model-plus-interface direction he described, and these
  screens are the natural surface for collecting its training data.
