#!/usr/bin/env python3
"""compare_scorers.py — old gates vs weight-of-evidence, on things that matter.

Offline, $0, no network, no key.

    python compare_scorers.py v11paternal2 e1

Counts alone cannot say which run is better: fewer identities might mean the
over-merging got worse, and more might mean real people were split. So this asks
questions with known right answers.

  1. THE TRANSATLANTIC MERGES. 24 identities in the delivered run join a Cuban
     parish to a Brazilian one ~6,600 km apart -- Maria x14, Antonio x10,
     "Reverendo Cura" -- and every one is wrong. A scorer that weighs location
     should destroy them. This counts how many survive in each run.

  2. DANIEL'S LABELS, as outcomes rather than scores.

  3. THE BIG CLERGY CLUSTERS. Daniel: clergy "can be merged very aggressively".
     Miguel Llopiz at 887 mentions is CORRECT, so a run that shatters him has
     over-corrected, and the naive reading -- "fewer huge clusters is tidier" --
     is exactly backwards.

  4. THE REVIEW RATE, against Daniel's ".1% is acceptable, 10% is not".
"""
import argparse
import collections
import json
import os


def load_ids(outdir, tag):
    p = os.path.join(outdir, f"{tag}.identities.json")
    if not os.path.exists(p):
        raise SystemExit(f"no identities for {tag!r} at {p}")
    return json.load(open(p, encoding="utf-8"))


def load_stats(outdir, tag):
    p = os.path.join(outdir, f"{tag}.stats.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def index(ids):
    out = {}
    for k, i in enumerate(ids):
        for m in i["mentions"]:
            out[(str(m.get("entry")), str(m.get("id")))] = k
    return out


def transatlantic(ids, geo):
    bad = []
    for i in ids:
        vols = sorted({str(m["entry"]).split("-")[0] for m in i["mentions"]})
        if len(vols) < 2:
            continue
        far = max((geo.km_between(a, b) or 0) for a in vols for b in vols if a < b)
        if far > 1000:
            bad.append((i.get("canonical_name"), i["n_mentions"], far))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--outdir", default="production/luna_v3/merge")
    ap.add_argument("--labels", default="labels.json")
    args = ap.parse_args(argv)

    from ssda_nlp_tools.volume_geo import load as load_geo
    geo = load_geo()

    runs = {}
    for tag in (args.old, args.new):
        ids = load_ids(args.outdir, tag)
        runs[tag] = {"ids": ids, "idx": index(ids), "stats": load_stats(args.outdir, tag)}

    a, b = runs[args.old], runs[args.new]
    if a["stats"].get("mentions") and b["stats"].get("mentions") and \
            a["stats"]["mentions"] != b["stats"]["mentions"]:
        raise SystemExit("different mention counts -- not comparable")

    print(f"{'':34s} {args.old:>14s} {args.new:>14s}")
    for k in ("mentions", "identities", "merged_identities", "auto_merges",
              "review_pairs"):
        va, vb = a["stats"].get(k), b["stats"].get(k)
        if va is None and vb is None:
            continue
        d = (vb - va) if (va is not None and vb is not None) else None
        print(f"  {k:32s} {va if va is not None else '-':>14} "
              f"{vb if vb is not None else '-':>14}"
              f"{'' if d is None else f'   {d:+,}'}")

    if geo:
        print("\n1. TRANSATLANTIC MERGES (Cuba joined to Brazil -- all wrong)")
        for tag in (args.old, args.new):
            t = transatlantic(runs[tag]["ids"], geo)
            print(f"   {tag:16s} {len(t):4d} identities span >1000 km")
            for nm, c, km in sorted(t, key=lambda r: -r[1])[:3]:
                print(f"        {str(nm)[:28]:30s} x{c:<3d} {km:5.0f} km")

    print("\n2. DANIEL'S CERTAIN LABELS")
    lab = [x for x in json.load(open(args.labels, encoding="utf-8"))["labels"]
           if x.get("likelihood") in (0, 100)]
    for tag in (args.old, args.new):
        idx = runs[tag]["idx"]
        ok = miss = 0
        for x in lab:
            ka = (str(x["a"]["entry"]), str(x["a"]["id"]))
            kb = (str(x["b"]["entry"]), str(x["b"]["id"]))
            if ka not in idx or kb not in idx:
                miss += 1
                continue
            ok += ((x["likelihood"] == 100) == (idx[ka] == idx[kb]))
        n = len(lab) - miss
        print(f"   {tag:16s} {ok}/{n} agree" + (f"  ({miss} unresolvable)" if miss else ""))

    print("\n3. LARGEST CLUSTERS (clergy recurrence is CORRECT, not over-merging)")
    for tag in (args.old, args.new):
        top = sorted(runs[tag]["ids"], key=lambda i: -i["n_mentions"])[:5]
        print(f"   {tag:16s} " + ", ".join(
            f"{str(i['canonical_name'])[:20]}({i['n_mentions']})" for i in top))

    print("\n4. REVIEW RATE   (Daniel: \".1% is acceptable, 10% is not\")")
    for tag in (args.old, args.new):
        s = runs[tag]["stats"]
        rv, au = s.get("review_pairs"), s.get("auto_merges")
        if rv is None:
            print(f"   {tag:16s} not recorded")
            continue
        ment = s.get("mentions") or 1
        print(f"   {tag:16s} {rv:,} review pairs, {au:,} auto "
              f"({rv / ment:.1f} review items per mention)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
