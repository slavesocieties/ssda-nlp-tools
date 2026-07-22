#!/usr/bin/env python3
"""Prepare Archivault administrative dossiers for deterministic QA and LLM review.

    python run_admin_pilot.py INPUT.json --out dossiers.json
"""
import argparse
import json
from pathlib import Path

from ssda_nlp_tools.admin_records import to_documents, to_page_index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page-index-out", help="optional deterministic, provenance-preserving page index")
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
    if args.page_index_out:
        index = to_page_index(result)
        Path(args.page_index_out).write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"page-index: {index['stats']['pages']} pages -> {args.page_index_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
