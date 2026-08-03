#!/usr/bin/env python3
"""run_merge.py — identity merging as a standalone, re-runnable stage.

Offline, $0, no network, no key.

Daniel, 2026-07-29: "This may be another reason to handle merging completely
separately from extraction."

He is right, and the reason is asymmetric cost. Extraction is the only step that
spends money ($24.53 for the current corpus) and it is settled: the records are
delivered and conformant. Merging spends nothing, is the step we understand
least, and is the one whose rules are still moving — the surname tiers in this
run are two days old and will change again once labelled data exists. Fusing the
two means every merge experiment looks like it needs a re-extraction, which is
both false and expensive enough to discourage the experiments.

So this stage takes delivered extraction output as a fixed input and owns
everything downstream of it. Re-running it with different thresholds costs
minutes and nothing else, and the extraction artifacts are never touched.

    python run_merge.py --tag v3
    python run_merge.py --tag strict --auto 0.90 --no-surname-tiers
    python run_merge.py --tag v3 --constraints labels.json    # feed review back in

Outputs, per --tag, under --outdir:
    identities.json     one row per resolved person
    review_queue.json   pairs the algorithm declined to decide
    stats.json          counts, and the config that produced them
"""
import argparse
import json
import os
import time

from ssda_nlp_tools.disambiguate import disambiguate_volume, format_disambiguation
from ssda_nlp_tools.likelihood_review_html import labels_to_constraints
from ssda_nlp_tools.review_html import decisions_to_constraints


def load_constraints(path):
    """Accept either review format. The binary page emits {"decisions": [...]},
    the likelihood page {"labels": [...]}; only 0% and 100% from the latter
    become constraints, which labels_to_constraints enforces."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "labels" in raw:
        return labels_to_constraints(raw), "likelihood"
    return decisions_to_constraints(raw), "binary"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled",
                    help="delivered extraction output; READ ONLY, never rewritten")
    ap.add_argument("--outdir", default="production/luna_v3/merge")
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--auto", type=float, default=0.86)
    ap.add_argument("--review", type=float, default=0.70)
    ap.add_argument("--year-window", type=int, default=60)
    ap.add_argument("--no-lifespan", action="store_true",
                    help="disable the chronology guard. For A/B measurement on a "
                         "FIXED corpus only -- comparing runs across different "
                         "corpora confounds the guard with the corpus.")
    ap.add_argument("--no-surname-tiers", action="store_true",
                    help="disable the tiered spelling bar (Daniel's Llopiz ruling)")
    ap.add_argument("--constraints", metavar="PATH",
                    help="decisions.json or labels.json from a review page")
    args = ap.parse_args(argv)

    paths = sorted(os.path.join(args.assembled, f)
                   for f in os.listdir(args.assembled)
                   if f.endswith(".materialized.json"))
    if not paths:
        ap.error(f"no *.materialized.json under {args.assembled}")

    entries = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        entries.extend(d.get("entries") or d.get("records") or [])

    constraints, kind = (None, None)
    if args.constraints:
        constraints, kind = load_constraints(args.constraints)
        print(f"constraints from {args.constraints} ({kind}): "
              f"{len(constraints['must'])} must, {len(constraints['cannot'])} cannot")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"{len(paths)} volumes, {len(entries)} entries")
    t0 = time.time()
    res = disambiguate_volume({"id": "corpus", "entries": entries},
                              auto_threshold=args.auto,
                              review_threshold=args.review,
                              year_window=args.year_window,
                              constraints=constraints,
                              surname_tiers=not args.no_surname_tiers,
                              volume_tag=args.tag)
    elapsed = time.time() - t0

    stats = dict(res["stats"])
    stats["config"] = {
        "auto_threshold": args.auto, "review_threshold": args.review,
        "year_window": args.year_window,
        "surname_tiers": not args.no_surname_tiers,
        "constraints": args.constraints, "constraints_kind": kind,
        "volumes": [os.path.basename(p) for p in paths],
        "entries": len(entries), "seconds": round(elapsed, 1),
    }

    base = os.path.join(args.outdir, args.tag)
    for name, payload in (("identities", res["identities"]),
                          ("review_queue", res["review_queue"]),
                          ("stats", stats)):
        with open(f"{base}.{name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    print(format_disambiguation(res, top=8))
    print(f"\n{elapsed:.0f}s -> {base}.{{identities,review_queue,stats}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
