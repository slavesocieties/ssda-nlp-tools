#!/usr/bin/env python3
"""Prepare Archivault administrative dossiers for deterministic QA and LLM review.

    python run_admin_pilot.py INPUT.json --out dossiers.json
"""
import argparse
import json
from pathlib import Path

from ssda_nlp_tools.admin_records import to_documents


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    source = Path(args.input)
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        ap.error("INPUT must be an Archivault JSON array")
    result = to_documents(items, source=str(source))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = result["stats"]
    print(f"documents: {s['documents']}; pages: {s['pages']}; markers omitted: {s['synthetic_markers_omitted']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
