# The first recall measurement — 2026-08-13

Daniel graded the 25 heavy rows. All 25 came back **0**, with the note: *"All are
very far from being even remotely possible matches."*

## The result

```
grades      : 45 of 200 rows, from both pages
point est.  : 0 true merges hidden in the blocked pool
represented : 2,022,533 of 3,013,215 pairs  (67.1%)
UNREPRESENTED : 990,682 pairs (32.9%) in strata with no graded row
```

**Zero estimated missed merges across two thirds of everything the pre-filter
discards.** That is the first recall figure this project has ever had, and it
supports the conservative design: the pairs being thrown away are, so far as a
historian can tell, genuinely different people.

What it does **not** say: the remaining 32.9% is *unrepresented*, not zero. And
12 strata standing for 113,217 pairs (3.8%) were sampled once each, so no
variance is estimable there and the whole of that mass rides on one judgement
apiece. The honest statement is **"zero, over the 67% we can speak for."**

## The bug that nearly made this a confident wrong answer

The heavy page baked each row's **page position** into its buttons —
`mk(0,…)`, `mk(1,…)` — and the download reads that dict. I had rewritten the
`data-i` attribute instead, verified `data-i`, and shipped. **No JavaScript reads
`data-i`.** I verified an attribute with no bearing on the output.

So the 25 grades came back keyed `0..24`. Those are page positions. They are
**also perfectly valid row numbers in the 200-row sample** — and they are the
exact rows Daniel had already graded.

Merging the two files would have:

- raised **no clash**, because both say `0` everywhere they overlap;
- passed **the lock**, because the sample file never moved;
- passed **the range check**, because 0..24 are legal row numbers;
- and attributed **1,945,945 pairs of evidence to the twenty lightest rows**,
  reporting a clean zero over 0.07% of the pool while looking exactly like a
  real result covering two thirds of it.

Caught because the returned indices were checked against
`heavy_rows.json` before estimating, and they did not match.

### Fixed in three places

1. **The builder** now rewrites the `mk(<i>,` index, and then **re-reads its own
   rendered HTML** and refuses to write a page whose onclick indices do not match
   the intended rows. Verifying the artifact, not the intent.
2. **The estimator** refuses any file tagged `heavy` whose indices are exactly
   `0..n-1` and which carries no `_remapped_from`, naming the remap needed.
3. **Daniel's returned file was remapped**, not re-requested. His judgements are
   correct and were never in question; only our key was wrong.

The lesson is narrow and worth keeping: **verifying the wrong attribute is not
verification.** The check ran, passed, and proved nothing, because it tested a
property the system does not consult.
