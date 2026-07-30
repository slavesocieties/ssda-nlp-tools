#!/usr/bin/env python3
"""analyze_labels.py — what Daniel's labels say about each merge rule.

Offline, $0, no network, no key. Run this the day `labels.json` comes back.

Every rule the matcher applies was written by us, mostly in the last few days,
and the pair sample was built to span all of them: six distinct ways a merge can
now be refused, plus auto-merges and sub-threshold pairs. So the labels are not
only training data for a future model, they are the first outside evidence about
whether any of those rules is wrong.

The asymmetry worth watching is that we have been tightening all week. Every
measurement so far could only show over-merging, because that is the failure a
too-loose rule produces and it is visible in the graph. A too-STRICT rule
produces silence: two records that should be one person simply stay apart, and
nothing about that looks wrong from the inside. Daniel labelling a refused pair
100% is the only way we find those, so `too_strict` below is the column that
carries new information.

    python analyze_labels.py labels.json
    python analyze_labels.py labels.json --pairs production/luna_v3/training_set/pairs_core.json
"""
import argparse
import json
from collections import Counter, defaultdict

# Daniel's scale. 100 and 0 are decisions; the middle is graded doubt.
SAME, PROBABLY_SAME, UNCLEAR, PROBABLY_DIFF, DIFF = 100, 75, 50, 25, 0


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merged_by_algorithm(disposition: str) -> bool:
    """Did the pipeline actually join these two? Everything that is not a plain
    `auto` is a refusal of some kind, including the sacrament and cluster guards."""
    return disposition == "auto"


def verdict(row):
    """Compare one human label against what the algorithm did.

    The middle of the scale is deliberately NOT counted as agreement or
    disagreement. 50% means the reviewer could not tell, and scoring that as
    either would manufacture a signal out of an admission of uncertainty.
    """
    v, disp = row.get("likelihood"), row.get("disposition") or "?"
    if v is None:
        return None
    merged = merged_by_algorithm(disp)
    if v in (SAME, PROBABLY_SAME):
        return "agree" if merged else "too_strict"
    if v in (DIFF, PROBABLY_DIFF):
        return "too_loose" if merged else "agree"
    return "unclear"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels.json downloaded from the review page")
    ap.add_argument("--pairs", default="production/luna_v3/training_set/pairs_core.json",
                    help="the sample it was generated from, for stratum weights")
    ap.add_argument("--min-per-rule", type=int, default=5,
                    help="below this many labels, report a rule as UNDER-SAMPLED "
                         "rather than quoting a rate from almost nothing")
    args = ap.parse_args(argv)

    rows = load(args.labels).get("labels") or []
    done = [r for r in rows if r.get("likelihood") is not None]
    if not done:
        print("no labelled pairs yet")
        return 0

    print(f"{len(done):,} of {len(rows):,} pairs labelled "
          f"({100*len(done)/len(rows):.0f}%)\n")

    dist = Counter(r["likelihood"] for r in done)
    print("label distribution")
    for v in (SAME, PROBABLY_SAME, UNCLEAR, PROBABLY_DIFF, DIFF):
        n = dist.get(v, 0)
        bar = "#" * round(40 * n / max(dist.values())) if dist else ""
        print(f"  {v:3d}%  {n:5d}  {bar}")
    decisive = dist.get(SAME, 0) + dist.get(DIFF, 0)
    print(f"\n  usable as hard constraints (0% or 100%): {decisive:,}")
    print(f"  graded, training signal only            : {len(done)-decisive:,}")

    # ---- per rule -------------------------------------------------------- #
    by_rule = defaultdict(Counter)
    for r in done:
        v = verdict(r)
        if v:
            by_rule[r.get("disposition") or "?"][v] += 1

    # widen to the longest rule name; `blocked-surname-tier-uninformative` is 34
    # characters and would otherwise push every column out of alignment
    width = max(34, max((len(k) for k in by_rule), default=34) + 2)
    print(f"\n{'rule / disposition':{width}s} {'n':>5s} {'agree':>7s} "
          f"{'too_strict':>11s} {'too_loose':>10s}")
    concerns = []
    for rule in sorted(by_rule, key=lambda k: -sum(by_rule[k].values())):
        c = by_rule[rule]
        n = sum(c.values())
        judged = n - c["unclear"]
        if n < args.min_per_rule:
            print(f"  {rule:{width-2}s} {n:5d}   UNDER-SAMPLED, no rate quoted")
            continue
        ts, tl = c["too_strict"], c["too_loose"]
        rate = f"{100*c['agree']/judged:6.0f}%" if judged else "     -"
        print(f"  {rule:{width-2}s} {n:5d} {rate:>7s} {ts:11d} {tl:10d}")
        if judged and (ts + tl) / judged > 0.20:
            concerns.append(((ts + tl) / judged, rule, ts, tl, judged))

    # ---- what to change -------------------------------------------------- #
    print("\nwhat the labels are saying")
    if not concerns:
        print("  no rule disagrees with Daniel on more than 20% of its judged pairs.")
    # worst disagreement first, not biggest sample: if several rules are
    # flagged, the one to fix is the one he overrules most often
    for _, rule, ts, tl, judged in sorted(concerns, reverse=True):
        direction = ("TOO STRICT: it is refusing merges he would make"
                     if ts > tl else
                     "TOO LOOSE: it is making merges he would not")
        print(f"  {rule}: {direction} "
              f"({ts} strict / {tl} loose of {judged} judged)")

    # a refusal rule that is never wrong in the strict direction is doing its
    # job; one that is often wrong is costing real links invisibly
    strict_total = sum(c["too_strict"] for c in by_rule.values())
    loose_total = sum(c["too_loose"] for c in by_rule.values())
    print(f"\n  overall: {strict_total} merges refused that he would make, "
          f"{loose_total} made that he would not")
    if strict_total > loose_total:
        print("  -> the tightening this week has overshot; loosen before training.")
    elif loose_total > strict_total:
        print("  -> still over-merging on balance; the model has room to help.")

    # ---- weighting ------------------------------------------------------- #
    try:
        pairs = load(args.pairs).get("pairs") or []
        w = {(p["a"]["entry"], p["a"]["id"], p["b"]["entry"], p["b"]["id"]):
             p.get("weight") for p in pairs}
        hit = [w[k] for r in done
               if (k := (r["a"]["entry"], r["a"]["id"],
                         r["b"]["entry"], r["b"]["id"])) in w and w[k]]
        if hit:
            print(f"\n  inverse-probability weights available for {len(hit):,} of "
                  f"{len(done):,} labelled pairs (min {min(hit):g}, max {max(hit):g})")
            print("  the sample over-represents rare case types on purpose; use "
                  "these to recover the true distribution when training.")
    except (OSError, json.JSONDecodeError, KeyError):
        print("\n  (sample file not readable; skipping weight check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
