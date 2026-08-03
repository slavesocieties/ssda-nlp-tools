#!/usr/bin/env python3
"""compare_merge_runs.py — A/B two merge runs, and refuse invalid comparisons.

Offline, $0, no network, no key.

    python compare_merge_runs.py v8control2 v8lifespan

WHY THIS REFUSES RATHER THAN REPORTS
------------------------------------
Two comparisons went wrong today in opposite directions, and both looked fine:

  CONFOUNDED   v7 vs v8 showed identities 22,801 -> 32,943, which I nearly
               attributed to a new guard. The corpus had grown from 5 volumes to
               7 underneath the comparison; mentions went 27,875 -> 39,697.

  NOT A CONTROL  v8control vs v8lifespan showed a difference of exactly zero on
               every metric, which reads as "the guard does nothing". The
               control's --no-lifespan flag set an argparse value that no code
               consumed, so both runs had the guard ON. `blocked-lifespan` was
               1,416 in BOTH.

The second is the more dangerous, because a null result invites you to stop
looking. So this checks the preconditions before it prints a single number:

  * same mention count      -- otherwise the corpora differ and nothing is
                               attributable to the change under test
  * same volume list        -- same reason, caught earlier
  * the toggle actually moved -- if the flag under test shows the same block
                               count in both runs, the control is not a control

Any of those failing is an error, not a footnote.
"""
import argparse
import json
import os
import sys

METRICS = ("mentions", "identities", "merged_identities", "auto_merges",
           "review_pairs", "pairs_blocked_by_context")


def load(outdir, tag):
    p = os.path.join(outdir, f"{tag}.stats.json")
    if not os.path.exists(p):
        raise SystemExit(f"no stats for {tag!r} at {p}")
    return json.load(open(p, encoding="utf-8"))


def blocks(stats):
    return stats.get("merges_blocked_by_surname_tier") or {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control")
    ap.add_argument("treatment")
    ap.add_argument("--outdir", default="production/luna_v3/merge")
    ap.add_argument("--toggle", default="blocked-lifespan",
                    help="the block label the treatment is supposed to add")
    ap.add_argument("--force", action="store_true",
                    help="print the comparison even if it is invalid")
    args = ap.parse_args(argv)

    A, B = load(args.outdir, args.control), load(args.outdir, args.treatment)
    ca, cb = A.get("config") or {}, B.get("config") or {}

    problems = []
    if A.get("mentions") != B.get("mentions"):
        problems.append(f"different mention counts ({A.get('mentions'):,} vs "
                        f"{B.get('mentions'):,}) -- the corpora are not the same, "
                        f"so nothing here is attributable to the change")
    if ca.get("volumes") != cb.get("volumes"):
        problems.append("different volume lists between runs")
    ta, tb = blocks(A).get(args.toggle, 0), blocks(B).get(args.toggle, 0)
    if ta == tb:
        problems.append(f"{args.toggle} is {ta:,} in BOTH runs -- the control is "
                        f"not a control, the toggle did not take effect")

    if problems:
        print(f"INVALID COMPARISON: {args.control} vs {args.treatment}\n")
        for p in problems:
            print(f"  * {p}")
        if not args.force:
            print("\nRefusing to print numbers that cannot be interpreted. "
                  "Fix the run, or pass --force if you know what you are doing.")
            return 1
        print("\n--force given; the figures below are NOT attributable.\n")

    print(f"A/B on {A.get('mentions', 0):,} mentions, "
          f"{len(ca.get('volumes') or [])} volumes\n")
    print(f"{'':30s} {'control':>13} {'treatment':>13} {'delta':>11}")
    for k in METRICS:
        if k not in A and k not in B:
            continue
        a, b = A.get(k, 0), B.get(k, 0)
        pct = f"{100 * (b - a) / a:+.2f}%" if a else ""
        print(f"  {k:28s} {a:13,} {b:13,} {b - a:+11,} {pct}")

    print(f"\n  {args.toggle}: {ta:,} (control) -> {tb:,} (treatment)")
    keys = sorted(set(blocks(A)) | set(blocks(B)))
    moved = [(k, blocks(A).get(k, 0), blocks(B).get(k, 0))
             for k in keys if blocks(A).get(k, 0) != blocks(B).get(k, 0)]
    if moved:
        print("  other block reasons that moved:")
        for k, a, b in moved:
            print(f"    {k:34s} {a:10,} -> {b:10,} ({b - a:+,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
