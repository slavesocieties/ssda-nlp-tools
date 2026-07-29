#!/usr/bin/env python3
"""Create an auditable review queue for ethnicity values outside vocab.json.

Ethnicity is historically open-ended.  This tool does not normalize, delete,
or guess at a value.  It preserves the emitted term and provides source-text
context so a domain reviewer can decide whether it is an accepted historical
descriptor, an origin, a phenotype/category error, or a transcription issue.
It is offline and makes no API calls.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ssda_nlp_tools import vocab as V


def compact(text: object, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def audit(paths: list[Path], sample_limit: int = 3, context_chars: int = 480) -> dict:
    v = V.load_vocab()
    terms: dict[str, dict] = {}
    known = Counter()
    seen = 0
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for record in doc.get("entries") or doc.get("records") or []:
            data = record.get("data") or {}
            for person in data.get("people") or []:
                value = person.get("ethnicity")
                if value in (None, "", []):
                    continue
                seen += 1
                value = str(value)
                if v.is_known("ethnicity", value):
                    known[value] += 1
                    continue
                item = terms.setdefault(value, {
                    "term": value, "count": 0,
                    "review_status": "unreviewed",
                    "rule": "Preserve verbatim; do not auto-map an open historical descriptor.",
                    "samples": [],
                })
                item["count"] += 1
                if len(item["samples"]) < sample_limit:
                    item["samples"].append({
                        "entry_id": record.get("id"),
                        "person": person.get("name"),
                        "faithful_context": compact(record.get("text_faithful"), context_chars),
                        "normalized_context": compact(record.get("text_normalized"), context_chars),
                    })
    values = sorted(terms.values(), key=lambda x: (-x["count"], x["term"].casefold()))
    return {
        "purpose": "Review queue only; no historical value has been altered.",
        "ethnicity_values_seen": seen,
        "known_value_instances": sum(known.values()),
        "unreviewed_value_instances": sum(x["count"] for x in values),
        "unreviewed_distinct_values": len(values),
        "terms": values,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Ethnicity Review Queue",
        "",
        report["purpose"],
        "",
        f"- Ethnicity values seen: {report['ethnicity_values_seen']}",
        f"- Known-vocabulary instances: {report['known_value_instances']}",
        f"- Unreviewed instances: {report['unreviewed_value_instances']}",
        f"- Distinct unreviewed terms: {report['unreviewed_distinct_values']}",
        "",
        "| Term | Count | Status |",
        "|---|---:|---|",
    ]
    lines += [f"| {x['term']} | {x['count']} | {x['review_status']} |" for x in report["terms"]]
    for item in report["terms"]:
        lines.extend(["", f"## {item['term']} ({item['count']})", "", item["rule"]])
        for sample in item["samples"]:
            lines.extend([
                "",
                f"- `{sample['entry_id']}` — {sample.get('person') or 'unnamed person'}",
                f"  - Faithful: {sample['faithful_context'] or '(none)'}",
                f"  - Normalized: {sample['normalized_context'] or '(none)'}",
            ])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", nargs="+", type=Path, help="materialized corpus JSON file(s)")
    ap.add_argument("--out", type=Path, required=True, help="audit JSON output")
    ap.add_argument("--markdown", type=Path, help="optional human-readable review queue")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args(argv)
    if args.samples < 1:
        ap.error("--samples must be positive")
    report = audit(args.records, args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    print(f"wrote {args.out}: {report['unreviewed_distinct_values']} terms / "
          f"{report['unreviewed_value_instances']} instances require review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
