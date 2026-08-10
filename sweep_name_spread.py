"""Can the name axis be made informative again without the transatlantic blowup?

FINDING THIS ANSWERS
--------------------
`MAX_NAME_LLR = 5.5` binds for 100% of full names and 91.7% of first-only names,
so the name term is a constant and carries no information. Measured on real
pairs the mean name contribution is 3.820 for full names and 3.855 for first-only
ones -- backwards, and flat. Daniel's ruling ("first name only, decades apart ->
very low; first and last name, within a few years -> moderately high") cannot be
satisfied by a model whose name term does not vary with the name.

WHY THE OBVIOUS FIX FAILED BEFORE
---------------------------------
Raising the cap to 7.5 added ~2 nats to EVERY pair uniformly, and transatlantic
false merges went from 56 to 1,169 mentions. The cap and the location weight are
coupled: lifting the floor lifts everything, including pairs a continent apart.

WHAT THIS SWEEPS INSTEAD
------------------------
Keep the floor exactly where it is and restore only the SPREAD, by compressing
the excess rarity rather than discarding it:

    llr = MAX_NAME_LLR + slope * (rarity - MAX_NAME_LLR)      rarity > cap

slope=0 is today's hard cap; slope=1 is no cap at all. Anything between buys
ordering at a known cost. The cost that matters is not the total merge count but
the transatlantic count, so both are reported, and the run is a measurement --
it changes no default.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats
from ssda_nlp_tools.textmatch import name_similarity, name_tokens, phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo


def build(assembled, volumes):
    entries = []
    for p in sorted(glob.glob(f"{assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {assembled!r}")
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    return M, NameStats(M, is_clergy=E._clergy), load_geo(volumes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--max-block", type=int, default=60)
    a = ap.parse_args(argv)

    M, stats, geo = build(a.assembled, a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    blocks = collections.defaultdict(list)
    for m in M:
        blocks[phonetic_key(m.get("name"))].append(m)

    # Score each pair ONCE into parts, then re-add the name term per slope.
    # Rescoring the whole corpus per slope would take hours; the name term is
    # the only thing that changes, so everything else is computed once.
    rows = []
    for key, g in blocks.items():
        if not key or len(g) < 2 or len(g) > a.max_block:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                x, y = g[i], g[j]
                if x["_entry"] == y["_entry"] or D.lifespan_conflict(x, y):
                    continue
                r = E.score(x, y, stats, geo=geo, vol_of=vol_of)
                if r["vetoed"]:
                    continue
                name_w = sum(w for lbl, w in r["terms"] if lbl.startswith("name"))
                rest = r["log_odds"] - name_w
                sim = name_similarity(x.get("name"), y.get("name"))
                rarity = min(-math.log(stats.p(x.get("name"))),
                             -math.log(stats.p(y.get("name"))))
                yx, yy = x.get("_year"), y.get("_year")
                gap = abs(yx - yy) if (yx and yy) else None
                full = (len(name_tokens(x.get("name")) or []) > 1 and
                        len(name_tokens(y.get("name")) or []) > 1)
                lvl = geo.same_place(vol_of(x), vol_of(y)) if geo else None
                rows.append((rest, sim, rarity, gap, full, lvl))
    print(f"{len(rows):,} scorable pairs\n")

    cap = E.MAX_NAME_LLR
    print(f"{'slope':>6} {'name term':>20} {'grid ordering':>32} "
          f"{'>=auto':>9} {'cross-country >=auto':>21}")
    print(f"{'':6} {'full / first':>20} {'full-close vs first-close':>32}")
    for slope in (0.0, 0.15, 0.25, 0.4, 0.6, 1.0):
        cells = collections.defaultdict(list)
        n_auto = n_auto_far = 0
        nm_full, nm_first = [], []
        for rest, sim, rarity, gap, full, lvl in rows:
            llr = cap + slope * (rarity - cap) if rarity > cap else rarity
            lo = rest + sim * llr
            (nm_full if full else nm_first).append(sim * llr)
            if lo >= E.AUTO_MERGE_LOG_ODDS:
                n_auto += 1
                if lvl in (None, "none", "country"):
                    n_auto_far += 1
            if gap is not None:
                tm = "close" if gap <= 5 else ("far" if gap >= 20 else "mid")
                cells[("full" if full else "first", tm)].append(
                    1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0)
        fc = statistics.median(cells[("full", "close")])
        ic = statistics.median(cells[("first", "close")])
        ff = statistics.median(cells[("full", "far")])
        ok = "OK" if (fc > ic and fc > ff) else "VIOLATED"
        print(f"{slope:6.2f} {statistics.fmean(nm_full):9.2f} /{statistics.fmean(nm_first):8.2f} "
              f"{fc:12.3f} vs {ic:7.3f}  {ok:>8s} {n_auto:9,} {n_auto_far:21,}")

    print("\nslope 0.00 is the current production model (hard cap).")
    print("'cross-country >=auto' is the transatlantic failure mode that the")
    print("cap-to-7.5 experiment blew up; it is the number to watch, not the total.")


if __name__ == "__main__":
    main()
