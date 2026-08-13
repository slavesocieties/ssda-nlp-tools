"""Check the scorer against Daniel's stated expectation, as a rank ordering.

Daniel, 2026-08-07: "that's a probabilistic calculation based on name rarity and
temporal proximity, and the answer is going to be very different for different
combinations of those factors. First name only, decades apart -> very low; first
and last name, within a few years -> moderately high."

That is a falsifiable claim about a 2x2. It does not pin down numbers, but it
does pin down an ORDER, and an order is testable:

    full name  + close  >  full name  + far
    full name  + close  >  first only + close
    first only + far    is the floor

Measured on real corpus pairs, not hand-built ones, because the question is what
the model does to the data Daniel is actually looking at. Nothing here is a pass
or fail of his ruling -- it is a check that the implementation agrees with it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, score
from ssda_nlp_tools.textmatch import name_tokens, phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--max-block", type=int, default=60)
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {a.assembled!r}")
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    blocks = collections.defaultdict(list)
    for m in M:
        blocks[phonetic_key(m.get("name"))].append(m)

    cells = collections.defaultdict(list)
    for key, g in blocks.items():
        if not key or len(g) < 2 or len(g) > a.max_block:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                x, y = g[i], g[j]
                if x["_entry"] == y["_entry"]:
                    continue
                if D.lifespan_conflict(x, y):
                    continue          # impossible pairs are not part of the claim
                yx, yy = x.get("_year"), y.get("_year")
                if not (yx and yy):
                    continue
                gap = abs(yx - yy)
                # "first name only" vs "first and last": both sides must carry a
                # surname for the pair to count as a full-name match.
                full = (len(name_tokens(x.get("name")) or []) > 1 and
                        len(name_tokens(y.get("name")) or []) > 1)
                r = score(x, y, stats, geo=geo, vol_of=vol_of)
                if r["vetoed"]:
                    continue
                nm = "full name" if full else "first only"
                tm = "within 5y" if gap <= 5 else ("decades apart" if gap >= 20
                                                   else "6-19y")
                cells[(nm, tm)].append(r["probability"])

    print("Daniel's grid, measured on real pairs (posterior probability of "
          "same person)\n")
    print(f"  {'':12s} {'within 5y':>22s} {'6-19y':>22s} {'decades apart':>22s}")
    grid = {}
    for nm in ("full name", "first only"):
        row = f"  {nm:12s}"
        for tm in ("within 5y", "6-19y", "decades apart"):
            v = cells.get((nm, tm)) or []
            if v:
                grid[(nm, tm)] = statistics.median(v)
                row += f" {statistics.median(v):8.3f} (n={len(v):>7,})"
            else:
                row += f" {'-':>22s}"
        print(row)

    print("\n  ordering claimed by Daniel:")
    checks = [
        ("full name+close  >  full name+decades",
         ("full name", "within 5y"), ("full name", "decades apart")),
        ("full name+close  >  first only+close",
         ("full name", "within 5y"), ("first only", "within 5y")),
        ("first only+close >  first only+decades",
         ("first only", "within 5y"), ("first only", "decades apart")),
    ]
    ok = True
    for label, hi, lo in checks:
        if hi not in grid or lo not in grid:
            print(f"    {label:44s} SKIP (empty cell)"); continue
        good = grid[hi] > grid[lo]
        ok &= good
        print(f"    {label:44s} {grid[hi]:.3f} vs {grid[lo]:.3f}  "
              f"{'OK' if good else '<-- VIOLATED'}")
    floor = min(grid, key=grid.get) if grid else None
    print(f"\n  lowest cell is {floor}, which should be "
          f"('first only', 'decades apart')")
    print(f"\n  ordering: {'consistent with the ruling' if ok else 'INCONSISTENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
