#!/usr/bin/env python3
"""run_eval.py — score extracted data against a gold set (or compare two runs).

    python run_eval.py GOLD.json PRED.json [--errors] [--json out.json] [--threshold 0.72]

GOLD / PRED may be either a training-data file ({"examples":[...]}) or a volume
record ({"entries":[...]}). With no gold labels you can still point GOLD at one
model's output and PRED at another to get an inter-model *agreement* report.
No API keys, no network — this only reads JSON.
"""
import argparse
import json
import sys

from ssda_nlp_tools.evaluate import evaluate, format_report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold", help="gold (or reference) JSON")
    ap.add_argument("pred", help="predicted (or comparison) JSON")
    ap.add_argument("--threshold", type=float, default=0.72,
                    help="name-match threshold for aligning people (default 0.72)")
    ap.add_argument("--errors", action="store_true", help="print per-entry error analysis")
    ap.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    args = ap.parse_args(argv)

    report = evaluate(args.gold, args.pred, name_threshold=args.threshold)
    print(format_report(report, show_errors=args.errors))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nfull report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
