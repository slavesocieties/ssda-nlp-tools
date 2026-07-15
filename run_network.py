#!/usr/bin/env python3
"""run_network.py — resolve people across a volume and build their social graph.

    python run_network.py VOLUME.json [--tag SSDA0013] [--graphml net.graphml]
                          [--json net.json] [--resolved resolved.json]
                          [--auto 0.86] [--review 0.70] [--top 12]

Runs disambiguation -> resolution -> network build in one shot. Exports GraphML
(loads in Gephi / networkx / Cytoscape) and/or a summary JSON. No API keys.
"""
import argparse
import json

from ssda_nlp_tools.resolve import resolve_volume
from ssda_nlp_tools.network import build_network, to_graphml, format_network


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--auto", type=float, default=0.86)
    ap.add_argument("--review", type=float, default=0.70)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--graphml", metavar="PATH", help="write GraphML")
    ap.add_argument("--json", metavar="PATH", help="write network summary JSON")
    ap.add_argument("--resolved", metavar="PATH", help="write the resolved volume + person index")
    args = ap.parse_args(argv)

    resolved = resolve_volume(args.volume, volume_tag=args.tag,
                              auto_threshold=args.auto, review_threshold=args.review)
    net = build_network(resolved)
    print(format_network(net, top=args.top))

    if args.graphml:
        to_graphml(net, args.graphml); print(f"\nGraphML  -> {args.graphml}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(net, f, ensure_ascii=False, indent=2)
        print(f"summary  -> {args.json}")
    if args.resolved:
        with open(args.resolved, "w", encoding="utf-8") as f:
            json.dump({"volume": resolved["volume"], "person_index": resolved["person_index"],
                       "review_queue": resolved["review_queue"]}, f, ensure_ascii=False, indent=2)
        print(f"resolved -> {args.resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
