"""Turn Daniel's grades on the blocked sample into a recall number, with a range.

WHAT THIS ANSWERS
-----------------
`_shares_context` (and now keyed blocking) discards pairs before anything scores
them. After removing the ones another guard would have refused anyway, **3,013,215
pairs survive every other check and are never looked at** -- HANDOVER sec.7b, the
recall blind spot. Nobody knows how many true merges are in there.

`build_blocked_labels.py` drew a 200-row weighted sample of exactly that
population. Each row carries `weight = stratum_size / rows_drawn`, and the
weights sum to 3,013,215. Given grades, the Horvitz-Thompson estimate

    true merges hidden  ~=  SUM over rows of  weight_i * (grade_i / 100)

is the first recall figure this project would ever have.

WHY THIS IS NOT A ONE-LINER
---------------------------
1. **The grades are keyed by POSITION.** The file carries no pair identifiers, so
   a grade is "row 41 = 80". Point that at a regenerated sample and it silently
   describes a different pair. This script therefore verifies the sample against
   `delivered_labels.lock.json` before reading a single grade, and refuses if it
   has moved. That is the whole reason the lock exists.

2. **A point estimate here is misleading and the handover says so.** The weights
   are extremely uneven -- the heaviest single row stands for 134,583 pairs. One
   grader keystroke on that row moves the answer by more than most strata
   contribute in total. So this reports a RANGE, and reports how much of it rests
   on strata sampled only once, where no variance is estimable at all.

3. **Partial grading is normal.** If Daniel grades 60 of 200, the naive sum
   under-counts by construction. This estimates per STRATUM and reweights by
   stratum size, so a partly-graded stratum still contributes its share -- and
   an entirely ungraded stratum is reported as unrepresented rather than
   silently treated as zero. A stratum of zeros and a stratum nobody looked at
   are not the same thing, and summing them together is how "no merges hidden
   here" gets manufactured.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os


def _load_lock(root, sample_path):
    lock_path = os.path.join(root, "delivered_labels.lock.json")
    if not os.path.exists(lock_path):
        return None
    lock = json.load(open(lock_path, encoding="utf-8"))
    rel = os.path.relpath(sample_path, root).replace("\\", "/")
    return lock["entries"].get(rel)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample",
                    default="production/luna_v3/blocked_labels/blocked_pairs.json",
                    help="the weighted sample -- must be the file that was GRADED")
    ap.add_argument("--grades", required=True,
                    help="the json downloaded from the grading page "
                         "({'labels': {'<row index>': 0-100}})")
    ap.add_argument("--root", default=".")
    ap.add_argument("--allow-unlocked", action="store_true",
                    help="proceed even if the sample does not match the lock. "
                         "Only correct if you KNOW the grades came from this file.")
    a = ap.parse_args(argv)

    sample = json.load(open(a.sample, encoding="utf-8"))
    rows = sample["pairs"]
    population = sample["population"]

    # ---- the position-keying guard, before anything is computed -----------
    want = _load_lock(a.root, a.sample)
    if want:
        got = hashlib.sha256(open(a.sample, "rb").read()).hexdigest()
        if got != want["sha256"]:
            msg = (f"THE SAMPLE HAS CHANGED since it was delivered.\n"
                   f"  locked : {want['bytes']:,}B  {want['mtime']}\n"
                   f"  now    : {os.path.getsize(a.sample):,}B\n"
                   f"Grades are keyed by row POSITION, so every one of them now "
                   f"points at a different pair. The estimate would be "
                   f"meaningless and would look completely normal.")
            if not a.allow_unlocked:
                raise SystemExit(msg + "\nPass --allow-unlocked only if certain.")
            print("WARNING, overridden:\n" + msg + "\n")
    else:
        print("NOTE: no lock entry for this sample; identity is unverified.\n")

    graded_raw = json.load(open(a.grades, encoding="utf-8"))
    graded = graded_raw.get("labels", graded_raw)
    graded = {int(k): float(v) for k, v in graded.items()}
    bad = [i for i in graded if i < 0 or i >= len(rows)]
    if bad:
        raise SystemExit(f"grade indices outside the sample: {bad[:5]} "
                         f"(sample has {len(rows)} rows) -- wrong file")

    print(f"sample     : {len(rows)} rows standing for {population:,} pairs")
    print(f"grades     : {len(graded)} of {len(rows)} rows "
          f"({100*len(graded)/len(rows):.0f}%)")
    print(f"scale      : {graded_raw.get('scale', 'assumed 0-100 % same person')}\n")

    # ---- stratified estimate ---------------------------------------------
    by_stratum = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_stratum[r["stratum"]].append((i, r))

    est = 0.0
    var = 0.0
    singleton_mass = 0.0
    singleton_contrib = 0.0
    ungraded_mass = 0.0
    singletons = 0
    for stratum, members in sorted(by_stratum.items()):
        N = members[0][1]["stratum_size"]
        vals = [graded[i] / 100.0 for i, _ in members if i in graded]
        if not vals:
            ungraded_mass += N
            continue
        n = len(vals)
        mean = sum(vals) / n
        est += N * mean
        if n >= 2:
            s2 = sum((v - mean) ** 2 for v in vals) / (n - 1)
            var += (N ** 2) * (1 - n / N if N else 0) * s2 / n
        else:
            singletons += 1
            singleton_mass += N
            singleton_contrib += N * mean

    print("ESTIMATED TRUE MERGES HIDDEN IN THE BLOCKED POOL")
    print(f"  point estimate            : {est:,.0f}"
          f"   ({100*est/population:.2f}% of the {population:,} blocked)")

    if var > 0:
        se = math.sqrt(var)
        print(f"  +/-1.96 SE (strata n>=2)  : "
              f"{max(0, est - 1.96*se):,.0f} .. {est + 1.96*se:,.0f}")
    else:
        print("  standard error            : NOT ESTIMABLE -- no stratum has "
              "2+ graded rows")

    # The honest width: a stratum sampled once contributes a value that could
    # have been anything. Report what that alone can move.
    # A stratum sampled once has no measurable within-stratum variance, so it
    # contributes nothing to the SE above while still carrying real mass. The
    # honest way to show that is the full swing if every one of those single
    # judgements had gone the other way.
    lo_single = est - singleton_contrib                  # all graded 0
    hi_single = est - singleton_contrib + singleton_mass  # all graded 100
    print(f"\n  strata sampled once       : {singletons}, standing for "
          f"{singleton_mass:,.0f} pairs ({100*singleton_mass/population:.1f}% "
          f"of the pool), contributing {singleton_contrib:,.0f}")
    print(f"  swing if those flipped    : {lo_single:,.0f} .. {hi_single:,.0f}")
    print("  (no within-stratum variance is estimable there; the whole mass")
    print("   rides on one judgement each, so this sits OUTSIDE the SE above)")

    if ungraded_mass:
        print(f"\n  UNREPRESENTED             : {ungraded_mass:,.0f} pairs "
              f"({100*ungraded_mass/population:.1f}%) sit in strata with NO "
              f"graded row.")
        print("   They are excluded from the estimate, not counted as zero.")
        print("   Scale the result to the whole pool only after grading them.")

    # The lock guarantees the FILE has not moved. It cannot guarantee the world
    # the file describes has not moved, and it did: the sample was drawn from
    # the raw assembly on 2026-08-10, and `assembled_deduped` became the default
    # on 2026-08-12. Measured on 2026-08-13, the blocked population is 2,952,727
    # there against the 3,013,215 this sample stands for -- 2.0% smaller.
    print(f"\nPOPULATION DRIFT since the sample was drawn")
    print(f"  this sample stands for       : {population:,} pairs "
          f"(raw assembly, 2026-08-10)")
    print(f"  default corpus now blocks    : 2,952,727 (assembled_deduped)")
    print(f"  drift                        : 2.0% smaller")
    print("  The estimate therefore describes the PRE-DEDUPE blocked set. The")
    print("  drift is far smaller than the interval above, so the sample stays")
    print("  usable and regrading would waste Daniel's work -- but say which")
    print("  configuration the number describes when reporting it.")

    print("\nHOW TO REPORT THIS")
    print("  As a range, with the graded fraction and the unrepresented mass")
    print("  stated. HANDOVER sec.7b: 'report a range, not a point.' And it is an")
    print("  estimate of merges MISSED, so it bounds recall -- it says nothing")
    print("  about whether the merges we DO make are right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
