#!/usr/bin/env python3
"""Turn validated OpenAI Batch JSONL output into one auditable SSDA volume.

Faithful segmented text, source images, and partial flags come exclusively from
the deterministic corpus.  Luna contributes only ``normalized`` and ``data``.
The command refuses incomplete, non-JSON, or non-normal-stop provider rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ssda_nlp_tools.batch_extract import parse_response


def response_results(paths):
    parsed, seen = {}, set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = row.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if row.get("response", {}).get("status_code") != 200 or len(choices) != 1 \
                    or choices[0].get("finish_reason") != "stop":
                raise ValueError(f"invalid provider response in {path.name}")
            text = choices[0].get("message", {}).get("content")
            if not isinstance(text, str):
                raise ValueError(f"missing response text in {path.name}")
            # Parse once without an expected list, then reject duplicate entry IDs.
            values, missing = parse_response(text, [], validate=True)
            if missing:
                raise ValueError(f"unexpected parser failure in {path.name}")
            overlap = set(values) & seen
            if overlap:
                raise ValueError(f"duplicate extracted entry IDs: {sorted(overlap)[:3]}")
            parsed.update(values)
            seen.update(values)
    return parsed


def materialize(corpus: dict, extracted: dict, allow_incomplete: bool = False) -> dict:
    entries = corpus.get("entries", [])
    canonical = {str(entry.get("id")): entry for entry in entries}
    missing = sorted(set(canonical) - set(extracted))
    extra = sorted(set(extracted) - set(canonical))
    if extra or (missing and not allow_incomplete):
        raise ValueError(f"corpus/extraction IDs differ; missing={missing[:5]} extra={extra[:5]}")
    out = []
    for entry in entries:
        eid = str(entry["id"])
        if eid not in extracted:
            continue
        model = extracted[eid]
        row = {
            "id": eid,
            "images": entry.get("images", []),
            "text_faithful": entry.get("text", ""),
            "normalized": model["normalized"],
            "data": model["data"],
        }
        if entry.get("partial"):
            row["partial"] = True
        out.append(row)
    return {"volume": str(corpus.get("volume", "")), "entries": out,
            "provenance": {"faithful_text": "deterministic segmented corpus",
                           "normalized_and_data": "validated GPT-5.6 Luna Batch output"},
            "coverage": {"corpus_records": len(entries), "materialized_records": len(out),
                         "missing_records": len(missing), "incomplete": bool(missing)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("output_glob", help="quoted glob of validated *.output.jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="materialize only validated returned records and mark coverage; never use for a final volume")
    args = ap.parse_args(argv)
    files = sorted(args.corpus.parent.parent.glob(args.output_glob))
    if not files:
        # The common invocation puts the glob directly below the output directory.
        files = sorted(Path().glob(args.output_glob))
    if not files:
        raise SystemExit("REFUSING: no provider output files matched")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    result = materialize(corpus, response_results(files), args.allow_incomplete)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"materialized {len(result['entries'])} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
