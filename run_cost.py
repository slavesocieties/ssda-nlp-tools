#!/usr/bin/env python3
"""run_cost.py — model the pipeline cost and find the recipe to hit a target.

    python run_cost.py [--repo .] [--target 0.01] [--min-shots 5]
                       [--pricing prices.json] [--images 300] [--json out.json]

Measures token counts from the repo's own extract.py / instructions.json /
training_data.json / a real volume, then reports the cost/image and the cheapest
quality-preserving recipe that gets transcription + normalization under --target.
No API keys, no network.
"""
import argparse
import json

from ssda_nlp_tools.cost import (
    Scenario, format_cost, format_quality_first, format_waterfall, lever_waterfall,
    load_pricing, measure_components, optimize, optimize_for_quality, scenario_cost)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--min-shots", type=int, default=5)
    ap.add_argument("--images", type=int, default=300, help="images/volume (cache horizon)")
    ap.add_argument("--pricing", default=None, help="JSON of model->{input,cached,output}")
    ap.add_argument("--metric", default="transcription_plus_normalization",
                    choices=["transcription_plus_normalization", "total",
                             "extraction", "transcription", "normalization"])
    ap.add_argument("--model", default="claude-haiku-4.5", help="extraction model for the waterfall")
    ap.add_argument("--corpus", type=int, default=750_000, help="corpus size for totals")
    ap.add_argument("--quality-first", action="store_true",
                    help="one-time-run mode: maximize output quality within --target "
                         "instead of minimizing cost within a quality floor. Never cuts "
                         "shots; sweeps vendor Batch API + prompt caching on every model.")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)

    pricing = load_pricing(args.pricing)
    comp = measure_components(args.repo)
    base = Scenario(images_per_volume=args.images)

    if args.quality_first:
        qreport = optimize_for_quality(comp, pricing, budget=args.target, base=base)
        print(format_quality_first(qreport))
        if args.json:
            slim = {k: v for k, v in qreport.items() if k != "all_ranked"}
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(slim, f, ensure_ascii=False, indent=2, default=str)
            print(f"\nreport -> {args.json}")
        return 0

    report = optimize(comp, pricing, target=args.target, min_shots=args.min_shots,
                      metric=args.metric, base=base)
    print(format_cost(report, comp))

    note = ("prices: verified 2026-07-16 against vendor docs, see "
            "eval_data/llm_model_research.md — override with --pricing; "
            "levers are robust to exact prices." if not args.pricing
            else f"prices: {args.pricing}")
    for m in dict.fromkeys([args.model, "gemini-2.5-flash", "claude-sonnet-5"]):
        if m in pricing:
            rows = lever_waterfall(comp, pricing, model=m, images_per_volume=args.images)
            print(format_waterfall(rows, pricing_note=note, corpus=args.corpus, model=m))

    if args.json:
        slim = {k: v for k, v in report.items() if k != "all_ranked"}
        slim["components"] = comp.__dict__
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nreport -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
