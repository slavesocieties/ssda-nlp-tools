#!/usr/bin/env python3
"""Create a deterministic, no-network SSDA routing manifest for one volume."""
import argparse
import json
from pathlib import Path

from ssda_nlp_tools.routing import route_volume
from ssda_nlp_tools.segment import load_pages


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Archivault JSON volume")
    ap.add_argument("--out", required=True, help="routing manifest JSON")
    ap.add_argument("--source-kind", choices=("auto", "sacramental", "administrative"), default="auto")
    args = ap.parse_args(argv)
    source = Path(args.input)
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        ap.error("INPUT must be an Archivault JSON array")
    result = route_volume(items, load_pages(str(source)), source=str(source), source_kind=args.source_kind)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"source kind: {result['source_kind']}; review required: {result['requires_review']}")
    print("routes: " + ", ".join(f"{k}={v}" for k, v in sorted(result["summary"].items())))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
