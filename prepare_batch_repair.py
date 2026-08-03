#!/usr/bin/env python3
"""Build a compact, offline repair batch from a failed validation report.

This tool performs no network calls and never reads an API key. It copies only
the rejected requests from the original compact batch file, retaining the
identical prompt header. Submit the result through ``run_luna_production.py``
with a new ``--run-id`` so the spend ledger can distinguish the repair from the
original request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_compact(path: Path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f"empty compact batch file: {path}")
    header = json.loads(lines[0]).get("header")
    rows = [json.loads(line) for line in lines[1:]]
    if not isinstance(header, dict) or not rows:
        raise ValueError(f"invalid compact batch file: {path}")
    return header, rows


def select_rejected_rows(rows, validation, receipt=None):
    rejected = validation.get("rejected_custom_ids") or {}
    if not isinstance(rejected, dict) or not rejected:
        raise ValueError("validation report has no rejected custom IDs")
    source_map = (receipt or {}).get("source_custom_ids") or {}
    wanted = {source_map.get(custom_id, custom_id) for custom_id in rejected}
    lookup = {row.get("custom_id"): row for row in rows}
    missing = sorted(wanted - set(lookup))
    if missing:
        raise ValueError(
            f"rejected IDs are absent from compact source: {missing[:5]}")
    return [row for row in rows if row.get("custom_id") in wanted]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--receipt", type=Path,
                        help="receipt containing run-id to source-id mappings")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise ValueError(f"refusing to overwrite existing repair file: {args.out}")
    header, rows = read_compact(args.compact)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    receipt = (json.loads(args.receipt.read_text(encoding="utf-8"))
               if args.receipt else None)
    selected = select_rejected_rows(rows, validation, receipt)
    repair_header = dict(header)
    repair_header["repair_of_validation"] = args.validation.name
    repair_header["repair_request_count"] = len(selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = [json.dumps({"header": repair_header}, ensure_ascii=False)]
    content.extend(json.dumps(row, ensure_ascii=False) for row in selected)
    args.out.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"prepared {len(selected)} rejected request(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"REFUSING: {exc}")
