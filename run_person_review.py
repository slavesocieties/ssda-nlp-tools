#!/usr/bin/env python3
"""run_person_review.py — one review screen per person, not per pair.

    python run_person_review.py VOLUME.json [MORE.json ...] --out person_review.html
                                [--limit 500] [--min-score 0.0] [--tag CORPUS]

Offline, $0, no network. The pairwise queue asks "are these two the same?" once
per pair (612,495 times on the current corpus). This asks "which of these, if
any, is this person?" once per person (13,967 screens, ~13 candidates each).

Decisions download in the SAME decisions.json shape the pairwise page produces,
so they feed straight back through:

    python run_review.py apply VOLUME.json decisions.json

--limit caps how many screens are written, ordered by best candidate score, so
the page stays openable and the highest-probability work comes first. The
uncapped corpus page would be hundreds of megabytes.
"""
import argparse
import json

from ssda_nlp_tools.disambiguate import disambiguate_volume
from ssda_nlp_tools.person_review import (format_summary, group_by_person,
                                          mention_to_identity, summarize)
from ssda_nlp_tools.person_review_html import render_person_review_html


def _load(paths):
    entries = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        entries.extend(d.get("entries") or d.get("examples") or [])
    return {"entries": entries}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volumes", nargs="+")
    ap.add_argument("--out", default="person_review.html")
    ap.add_argument("--tag", default="corpus")
    ap.add_argument("--limit", type=int, default=500,
                    help="screens to render, by best candidate score (0 = all)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--auto", type=float, default=0.86)
    ap.add_argument("--review", type=float, default=0.70)
    ap.add_argument("--no-block", action="store_true",
                    help="disable contextual pre-filtering (much slower)")
    ap.add_argument("--json", metavar="PATH", help="also write the screens as JSON")
    args = ap.parse_args(argv)

    res = disambiguate_volume(_load(args.volumes), auto_threshold=args.auto,
                              review_threshold=args.review, volume_tag=args.tag,
                              block_context=not args.no_block)
    identity_of = mention_to_identity(res["identities"])
    screens = group_by_person(res["review_queue"], min_score=args.min_score,
                              identity_of=identity_of)
    rep = summarize(screens, total_identities=res["stats"]["identities"],
                    total_pairs=len(res["review_queue"]))
    print(format_summary(rep))

    render_person_review_html(screens, args.out, tag=args.tag, limit=args.limit)
    shown = min(len(screens), args.limit) if args.limit else len(screens)
    print(f"\nrendered {shown:,} of {len(screens):,} screens -> {args.out}")
    if shown < len(screens):
        print(f"  ({len(screens) - shown:,} screens NOT rendered; raise --limit or "
              f"work through these first — they carry the strongest candidates)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": rep, "screens": screens}, f, ensure_ascii=False)
        print(f"  screens JSON -> {args.json}")
    print("decide, download decisions.json, then: run_review.py apply VOLUME.json decisions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
