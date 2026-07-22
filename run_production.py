#!/usr/bin/env python3
"""run_production.py — the free ($0) deterministic production build for the six
Drive volumes: segment every sacramental volume, tag each record with its
routing disposition, and emit production-ready record sets plus the corpus files
that `run_corpus_prompts.py` prices into the (paid, separately approved) Luna
extraction batches.

    python run_production.py [--manifests drive_pilots/routing_manifests]
                             [--outdir production]

No API calls, no spend. Reads the routing manifests (which point at each source
volume) and writes:

    production/segmented/<vol>.json   every segmented record + stats
    production/records/<vol>.json     records tagged by disposition
    production/corpus/<vol>.segmented.json   production records only, for staging
    production/production_summary.json       corpus-wide rollup

A record is `production` iff every page it spans routed
`deterministic-sacramental`; a record touching a fallback / re-transcribe /
index page is tagged for that path instead and withheld from the free output.
Then price the paid step (still $0, no send):

    python run_corpus_prompts.py --corpus production/corpus \\
        --outdir production/batches --model gpt-5.6-luna
"""
import argparse
import glob
import json
import os
from collections import Counter

from ssda_nlp_tools.segment import load_pages, segment_volume, to_canonical


def disposition(images, img_route):
    """Route a record from the routes of the pages it spans (worst wins)."""
    routes = {img_route.get(i, "unknown") for i in (images or [])}
    if routes <= {"deterministic-sacramental"}:
        return "production"
    if "retranscribe" in routes:
        return "blocked-retranscribe"
    if "luna-sacramental-fallback" in routes:
        return "needs-fallback"
    if "skip-index" in routes:
        return "index-context"
    return "review"


def build(manifests_dir, outdir):
    for sub in ("segmented", "records", "corpus"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    grand = Counter()
    summary = {"volumes": [], "totals": {}}
    for man_path in sorted(glob.glob(os.path.join(manifests_dir, "*.json"))):
        man = json.load(open(man_path, encoding="utf-8"))
        vol = os.path.basename(man_path).replace(".json", "")
        if man.get("source_kind") != "sacramental":
            summary["volumes"].append({"volume": vol, "source_kind": man.get("source_kind"),
                                       "handling": "administrative — not segmented here"})
            continue
        pages = load_pages(man["source"])
        res = segment_volume(pages)
        canon = to_canonical(res["entries"])
        json.dump({"volume": vol, "source": man["source"], "records": canon,
                   "stats": res["stats"]},
                  open(os.path.join(outdir, "segmented", f"{vol}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

        img_route = {r["source_image"]: r["route"] for r in man["routes"]}
        disp = Counter()
        prod = []
        for rec in canon:
            d = disposition(rec.get("images"), img_route)
            rec["disposition"] = d
            disp[d] += 1
            grand[d] += 1
            if d == "production":
                prod.append(rec)
        json.dump({"volume": vol, "source": man["source"], "production_records": prod,
                   "all_records": canon, "disposition_counts": dict(disp)},
                  open(os.path.join(outdir, "records", f"{vol}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        # corpus file for pricing/staging (production records only)
        json.dump({"volume": vol, "entries": [
                   {"id": r["id"], "text": r["text"], "partial": bool(r.get("partial"))}
                   for r in prod]},
                  open(os.path.join(outdir, "corpus", f"{vol}.segmented.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        summary["volumes"].append({"volume": vol, "pages": len(pages),
                                   "records": len(canon), "production": len(prod),
                                   "dispositions": dict(disp)})
        print(f"{vol}: {len(pages)} pages -> {len(canon)} records; production={len(prod)} "
              f"{dict(disp)}")
    summary["totals"] = dict(grand)
    summary["totals"]["records"] = sum(grand.values())
    json.dump(summary, open(os.path.join(outdir, "production_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n=== production totals ===")
    for k, v in grand.most_common():
        print(f"  {k}: {v}")
    print(f"  TOTAL records: {sum(grand.values())}")
    print(f"-> {outdir}/production_summary.json")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", default="drive_pilots/routing_manifests")
    ap.add_argument("--outdir", default="production")
    args = ap.parse_args(argv)
    build(args.manifests, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
