#!/usr/bin/env python3
"""run_qa.py — data-quality report for one or more extraction outputs.

    python run_qa.py FILE.json [MORE.json ...] [--json out.json] [--dup 0.75]

Flags suspected duplicate entries (window-overlap re-transcriptions), chronology
breaks, impossible dates, dangling references, malformed events, and attribute
vocabulary drift. No API keys, no network.
"""
import argparse
import json

from ssda_nlp_tools.qa import qa_volume, format_qa


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dup", type=float, default=0.75, help="duplicate text-sim threshold")
    ap.add_argument("--json", metavar="PATH", help="write all reports as JSON")
    args = ap.parse_args(argv)

    reports = []
    for f in args.files:
        rep = qa_volume(f, dup_threshold=args.dup)
        rep["file"] = f
        reports.append(rep)
        print(format_qa(rep))
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=2)
        print(f"reports -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
