#!/usr/bin/env python3
"""run_review.py — the human-in-the-loop cycle for identity merges.

Generate a review page from a volume's borderline pairs:

    python run_review.py make VOLUME.json --html review.html [--tag SSDA0013]

Open review.html in any browser (no server needed), press s/d/u per pair, click
"Download decisions.json", then fold the decisions back in:

    python run_review.py apply VOLUME.json decisions.json \\
        --graphml net.graphml --resolved resolved.json [--tag SSDA0013]

`apply` re-runs disambiguation with the decisions as must/cannot constraints and
rebuilds the resolved volume + network. No API keys, no network.
"""
import argparse
import json

from ssda_nlp_tools.disambiguate import disambiguate_volume, format_disambiguation
from ssda_nlp_tools.network import build_network, format_network, to_graphml
from ssda_nlp_tools.resolve import resolve_volume
from ssda_nlp_tools.review_html import render_review_html, decisions_to_constraints


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="generate the review HTML page")
    mk.add_argument("volume")
    mk.add_argument("--html", default="review.html")
    mk.add_argument("--tag", default=None)
    mk.add_argument("--auto", type=float, default=0.86)
    mk.add_argument("--review", type=float, default=0.70)

    app = sub.add_parser("apply", help="apply decisions.json and rebuild outputs")
    app.add_argument("volume")
    app.add_argument("decisions")
    app.add_argument("--tag", default=None)
    app.add_argument("--auto", type=float, default=0.86)
    app.add_argument("--review", type=float, default=0.70)
    app.add_argument("--graphml", metavar="PATH")
    app.add_argument("--resolved", metavar="PATH")

    args = ap.parse_args(argv)

    if args.cmd == "make":
        res = disambiguate_volume(args.volume, auto_threshold=args.auto,
                                  review_threshold=args.review, volume_tag=args.tag)
        render_review_html(res["review_queue"], args.html, tag=args.tag or "volume")
        print(f"{len(res['review_queue'])} borderline pairs -> {args.html}")
        print("open it in a browser, decide pairs, download decisions.json, then run apply.")
        return 0

    constraints = decisions_to_constraints(args.decisions)
    print(f"applying {len(constraints['must'])} must-link and "
          f"{len(constraints['cannot'])} cannot-link decisions")
    disamb = disambiguate_volume(args.volume, auto_threshold=args.auto,
                                 review_threshold=args.review, volume_tag=args.tag,
                                 constraints=constraints)
    print(format_disambiguation(disamb, top=8))
    resolved = resolve_volume(args.volume, disamb=disamb)
    net = build_network(resolved)
    print(format_network(net, top=8))
    if args.graphml:
        to_graphml(net, args.graphml); print(f"GraphML  -> {args.graphml}")
    if args.resolved:
        with open(args.resolved, "w", encoding="utf-8") as f:
            json.dump({"volume": resolved["volume"], "person_index": resolved["person_index"],
                       "review_queue": resolved["review_queue"]}, f, ensure_ascii=False, indent=2)
        print(f"resolved -> {args.resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
