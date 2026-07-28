#!/usr/bin/env python3
"""Build one-entry, non-overlapping Luna repair requests from a validation report.

The source compact batch file and raw provider output remain immutable.  This
tool only reads the request-level rejection list produced by
``run_luna_production.py`` and emits a new compact file.  Each repair request
contains exactly one original source entry, which bounds completion length and
prevents a single troublesome page from invalidating its neighbours.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_luna_production import expected_entries, read_compact, write_json


def entry_payload(row: dict) -> list[dict]:
    return json.loads(row["tail_message"]["content"])["entries"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="original compact batch input")
    ap.add_argument("validation", type=Path, help="invalid job's validation JSON")
    ap.add_argument("receipt", type=Path, help="invalid job's local receipt JSON")
    ap.add_argument("--label", required=True,
                    help="unique repair namespace, for example v3-repair1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-rejected", nargs="*", metavar="CUSTOM_ID",
                    help="repair only these rejected request IDs; each still expands "
                         "to its complete original request coverage")
    args = ap.parse_args(argv)

    header, rows = read_compact(args.source)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rejected = validation.get("rejected_custom_ids")
    if validation.get("valid") or not isinstance(rejected, dict) or not rejected:
        raise ValueError("validation must be an invalid guarded report with rejected_custom_ids")
    requested_ids = set(args.only_rejected or rejected)
    unknown = requested_ids - set(rejected)
    if unknown:
        raise ValueError(f"--only-rejected contains IDs not rejected by validation: {sorted(unknown)[:3]}")
    source_rows = {row["custom_id"]: row for row in rows}
    source_aliases = receipt.get("source_custom_ids", {})

    repair_rows = []
    repaired_sources = []
    for request_id in sorted(requested_ids):
        source_id = source_aliases.get(request_id, request_id)
        row = source_rows.get(source_id)
        if row is None:
            raise ValueError(f"rejected request {request_id} has no source row")
        repaired_sources.append({"request_id": request_id, "source_custom_id": source_id,
                                 "reason": rejected[request_id]})
        for item in entry_payload(row):
            entry_id = str(item["entry"])
            repair_rows.append({
                "custom_id": f"{args.label}-{entry_id}",
                "tail_message": {"role": "user", "content": json.dumps({
                    "instruction": (
                        "Process exactly ONE entry. Return exactly one results element whose "
                        f"entry is exactly {entry_id!r}. Never emit a continuation, a new ID, "
                        "or any entry not supplied below."),
                    "entries": [item],
                }, ensure_ascii=False)},
            })

    seen = set()
    duplicates = [row["custom_id"] for row in repair_rows
                  if row["custom_id"] in seen or seen.add(row["custom_id"])]
    if duplicates:
        raise ValueError(f"repair custom IDs are not unique: {duplicates[:3]}")
    if not repair_rows:
        raise ValueError("repair set is empty")

    repair_header = {**header, "volume": f"{header.get('volume', 'corpus')}-{args.label}",
                     "repair_label": args.label, "single_entry_repairs": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(
        [json.dumps({"header": repair_header}, ensure_ascii=False)]
        + [json.dumps(row, ensure_ascii=False) for row in repair_rows]) + "\n", encoding="utf-8")
    manifest = {"repair_file": str(args.out), "repair_label": args.label,
                "failed_request_count": len(repaired_sources),
                "repair_request_count": len(repair_rows),
                "repaired_sources": repaired_sources}
    write_json(args.out.with_suffix(".manifest.json"), manifest)
    print(f"built {len(repair_rows)} one-entry repairs from {len(repaired_sources)} rejected requests")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        raise SystemExit(2)
