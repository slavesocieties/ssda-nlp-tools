#!/usr/bin/env python3
"""build_training_sample.py — the disambiguation training set Daniel asked for.

Offline, $0, no network, no key. Scores every candidate pair across the corpus,
draws a variety-maximising stratified sample, and writes a labelling page on the
0/25/50/75/100 likelihood scale.

    python build_training_sample.py --size 2000 --tag core
    python build_training_sample.py --size 70000 --tag full10pct --no-text

Two tiers are worth building, and the reason is arithmetic rather than
preference. A literal 10% of the 701,238-pair review queue is ~70,000 decisions;
at a brisk 10 seconds each that is about 195 hours of Daniel's time. The
stratified draw is what makes a small sample viable: ~2,000 pairs already covers
every non-empty case type, because coverage is bounded by the number of strata
(a few hundred), not by the size of the population. Both tiers are produced from
the same reservoir and the same seed, so the small one is a strict subset in
character and the large one can be generated later without rescoring.
"""
import argparse
import json
import os
import pickle
import time

from ssda_nlp_tools.disambiguate import disambiguate_volume
from ssda_nlp_tools.likelihood_review_html import render_likelihood_review_html
from ssda_nlp_tools.training_sample import (STRATUM_AXES, StratifiedReservoir,
                                            attach_entry_text)


def load_corpus(paths):
    """Merge the per-volume materialized files into one volume so that
    cross-volume pairs are scored. Those are a case type of their own and they
    do not exist in a per-volume run."""
    entries, texts = [], {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        for e in d.get("entries") or d.get("records") or []:
            entries.append(e)
            t = e.get("text") or e.get("normalized") or e.get("faithful")
            if t:
                texts[str(e.get("id"))] = t
    return {"id": "corpus", "entries": entries}, texts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled",
                    help="directory of *.materialized.json volumes")
    ap.add_argument("--outdir", default="production/luna_v3/training_set")
    ap.add_argument("--tag", default="core")
    ap.add_argument("--size", type=int, default=2000, help="pairs to draw")
    ap.add_argument("--per-cell", type=int, default=120,
                    help="reservoir depth per stratum")
    ap.add_argument("--floor", type=float, default=0.30,
                    help="do not even log pairs below this score")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--rescore", action="store_true",
                    help="ignore the cached scoring pass and redo it")
    ap.add_argument("--html-limit", type=int, default=2500)
    ap.add_argument("--no-text", action="store_true",
                    help="omit register text (much smaller, much less reviewable)")
    args = ap.parse_args(argv)

    paths = sorted(os.path.join(args.assembled, f)
                   for f in os.listdir(args.assembled)
                   if f.endswith(".materialized.json"))
    if not paths:
        ap.error(f"no *.materialized.json under {args.assembled}")
    os.makedirs(args.outdir, exist_ok=True)

    volume, texts = load_corpus(paths)
    print(f"{len(paths)} volumes, {len(volume['entries'])} entries")

    # Scoring 7.3M pairs takes ~8.5 minutes and does not depend on --size, so a
    # size sweep would otherwise pay for it repeatedly. The cache key includes
    # every input that changes what gets scored; --size is deliberately not one.
    cache = os.path.join(args.outdir, "_reservoir.pkl")
    key = {"paths": paths, "floor": args.floor, "seed": args.seed,
           "per_cell": args.per_cell, "axes": list(STRATUM_AXES)}
    res = stats = None
    if os.path.exists(cache) and not args.rescore:
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        if blob.get("key") == key:
            res, stats = blob["res"], blob["stats"]
            print(f"reusing cached scoring: {res.total:,} pairs, "
                  f"{len(res.cells)} strata  (--rescore to redo)")
        else:
            print("cache is stale (inputs changed); rescoring")

    if res is None:
        res = StratifiedReservoir(per_cell=args.per_cell, seed=args.seed)
        t0 = time.time()
        # collect_review=False: the pair log already carries every review item
        # with its disposition, so materialising the ~1.1M-entry review queue as
        # well is gigabytes spent to build a list this pass never reads.
        stats = disambiguate_volume(volume, volume_tag="corpus",
                                    pair_log=res, pair_log_floor=args.floor,
                                    collect_review=False)["stats"]
        print(f"scored {res.total:,} pairs (>= {args.floor}) in "
              f"{time.time()-t0:.0f}s; {len(res.cells)} strata")
        with open(cache, "wb") as f:
            pickle.dump({"key": key, "res": res, "stats": stats}, f)

    drawn = res.draw(args.size)
    if not args.no_text:
        attach_entry_text(drawn, texts)
    cov = res.coverage(drawn)
    cov["disambiguation_stats"] = stats
    cov["seed"] = args.seed
    cov["floor"] = args.floor

    base = os.path.join(args.outdir, f"pairs_{args.tag}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "scale": "likelihood_same_percent",
                   "coverage": cov, "pairs": drawn}, f, ensure_ascii=False, indent=1)
    with open(base + ".coverage.json", "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=2)
    html = render_likelihood_review_html(drawn, base + ".html", tag=args.tag,
                                         limit=args.html_limit)

    print(f"\ndrew {len(drawn):,} pairs covering {cov['strata_represented']}"
          f"/{cov['strata_present']} strata "
          f"({cov['singleton_strata']} seen only once)")
    for axis, counts in cov["by_axis"].items():
        print(f"  {axis:14s} {counts}")
    if len(drawn) > args.html_limit:
        print(f"  NOTE: page shows the top {args.html_limit:,} by score; "
              f"the JSON holds all {len(drawn):,}")
    print(f"\n-> {base}.json\n-> {base}.coverage.json\n-> {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
