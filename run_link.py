#!/usr/bin/env python3
"""run_link.py — link people across multiple extraction outputs (chunks/volumes).

    python run_link.py FILE1.json FILE2.json [...] [--tags a b c] [--tag LINKED]
                       [--graphml net.graphml] [--registry registry.json]
                       [--auto 0.86] [--review 0.70] [--top 12]

Combines the inputs into one identity space, links recurring people across
files, and builds the unified social graph. No API keys, no network.
"""
import argparse
import json

from ssda_nlp_tools.link import link_volumes, format_link
from ssda_nlp_tools.network import to_graphml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="two or more extraction JSONs")
    ap.add_argument("--tags", nargs="*", default=None, help="short chunk labels")
    ap.add_argument("--tag", default="LINKED", help="prefix for global person ids")
    ap.add_argument("--auto", type=float, default=0.86)
    ap.add_argument("--review", type=float, default=0.70)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--graphml", metavar="PATH")
    ap.add_argument("--registry", metavar="PATH", help="write the person registry JSON")
    args = ap.parse_args(argv)

    res = link_volumes(args.files, tags=args.tags, volume_tag=args.tag,
                       auto_threshold=args.auto, review_threshold=args.review)
    print(format_link(res, top=args.top))

    if args.graphml:
        to_graphml(res["network"], args.graphml)
        print(f"\nGraphML  -> {args.graphml}")
    if args.registry:
        with open(args.registry, "w", encoding="utf-8") as f:
            json.dump({"registry": res["registry"], "review_queue": res["review_queue"],
                       "stats": res["stats"]}, f, ensure_ascii=False, indent=2)
        print(f"registry -> {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
