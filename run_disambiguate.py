#!/usr/bin/env python3
"""run_disambiguate.py — cluster person mentions across a volume into identities.

    python run_disambiguate.py VOLUME.json [--tag SSDA0013] [--out identities.json]
                               [--auto 0.86] [--review 0.70] [--top 15]

VOLUME.json may be a training-data file ({"examples":[...]}) or a volume record
({"entries":[...]}), each entry carrying data.people. Auto-merges confident
matches; routes borderline pairs to a ranked review queue. No API keys, no network.
"""
import argparse
import json

from ssda_nlp_tools.disambiguate import disambiguate_volume, format_disambiguation


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume", help="volume JSON with per-entry data.people")
    ap.add_argument("--tag", default=None, help="prefix for generated person ids")
    ap.add_argument("--auto", type=float, default=0.86, help="auto-merge threshold")
    ap.add_argument("--review", type=float, default=0.70, help="review-queue threshold")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    ap.add_argument("--out", metavar="PATH", help="write full identities+review JSON")
    args = ap.parse_args(argv)

    res = disambiguate_volume(args.volume, auto_threshold=args.auto,
                              review_threshold=args.review, volume_tag=args.tag)
    print(format_disambiguation(res, top=args.top))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nfull result -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
