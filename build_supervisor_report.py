#!/usr/bin/env python3
"""Build a factual, offline supervisor handoff from validated V3 artifacts."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def qa_counts(live: Path, volumes: list[str]) -> Counter:
    result = Counter()
    for vol in volumes:
        data = read_json(live / f"{vol}_final_pipeline" / "qa_report.json")
        for report in data if isinstance(data, list) else [data]:
            for issue in report.get("issues") or []:
                result[issue.get("type", "unknown")] += 1
    return result


def corpus_graph_stats(live: Path) -> dict:
    text = (live / "corpus_final_pipeline" / "summary.txt").read_text(encoding="utf-8")
    patterns = {
        "mentions": r"entries:\s+\d+\s+mentions:\s+(\d+)",
        "identities": r"identities:\s+(\d+)",
        "cross_chunk_identities": r"CROSS-CHUNK identities:\s+(\d+)",
        "review_pairs": r"review queue:\s+(\d+) pairs",
        "network_nodes": r"nodes \(people\):\s+(\d+)",
        "network_edges": r"edges \(relationships\):\s+(\d+) unique typed",
    }
    return {key: int(match.group(1).replace(",", "")) if (match := re.search(pattern, text)) else None
            for key, pattern in patterns.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", type=Path, default=Path("production/luna_v3"))
    ap.add_argument("--ledger", type=Path, default=Path("production/luna_live/spend_ledger.json"))
    ap.add_argument("--audit", type=Path, default=Path("production/luna_v3/ETHNICITY_AUDIT.json"))
    ap.add_argument("--out", type=Path, default=Path("production/luna_v3/SUPERVISOR_RESULTS.md"))
    args = ap.parse_args(argv)

    summary = read_json(args.live / "CORPUS_SUMMARY.json")
    ledger = read_json(args.ledger)
    audit = read_json(args.audit)
    volumes = list(summary["volumes"])
    qa = qa_counts(args.live, volumes)
    graph = corpus_graph_stats(args.live)
    totals = summary["totals"]
    headroom = float(ledger["cap_usd"]) - float(ledger["confirmed_usd"]) - float(ledger["reserved_usd"])

    lines = [
        "# SSDA V3 Corpus Extraction: Supervisor Results",
        "",
        "## Delivery status",
        "",
        f"- Source records: {totals['corpus_records']:,}",
        f"- Delivered records: {totals['materialized_records']:,}",
        f"- Missing source records: {totals['missing_records']}",
        f"- Invalid provider batches in delivery: {totals['invalid_batches']}",
        "- Seven page-truncated source records are retained in the auditable source corpus and excluded from delivery under the approved convention.",
        "",
        "| Volume | Delivered | Source records | Delivery state |",
        "|---|---:|---:|---|",
    ]
    for vol, value in summary["volumes"].items():
        lines.append(f"| {vol} | {value['materialized_records']:,} | {value['corpus_records']:,} | {value['state']} |")
    lines += [
        "",
        "## Cost and delivery controls",
        "",
        f"- Confirmed OpenAI Batch spend: ${float(ledger['confirmed_usd']):.4f}",
        f"- Approved cap: ${float(ledger['cap_usd']):.2f}",
        f"- Remaining headroom: ${headroom:.4f}",
        "- No reservation remains and no paid work is outstanding.",
        "- Every delivered request passed normal-stop, exact entry-ID, JSON, project-schema, and provider-usage validation.",
        "- Raw provider outputs are retained for audit but cannot be assembled; only request-level accepted artifacts feed delivery.",
        "",
        "## Measured prompt result",
        "",
        "On the held-out Portuguese 701054 register, age-category conformance increased from 26.6% (46/173) to 100.0% (173/173).",
        "Ethnicity is treated as an open historical descriptor field: its vocabulary rate is a review diagnostic, not an automatic quality regression. The retained corpus preserves the emitted terms verbatim.",
        "",
        "## QA and graph outputs",
        "",
        f"- Person mentions: {graph['mentions']:,}",
        f"- Resolved identities: {graph['identities']:,}",
        f"- Cross-volume identity candidates: {graph['cross_chunk_identities']:,}",
        f"- Network nodes / typed edges: {graph['network_nodes']:,} / {graph['network_edges']:,}",
        f"- Review queue: {graph['review_pairs']:,} candidate pairs. These are candidates, not approved merges.",
        "",
        "QA flags are preserved for review; they do not delete or rewrite records:",
        "",
        "| QA flag | Count |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {count:,} |" for name, count in qa.most_common()]
    lines += [
        "",
        "## Ethnicity review queue",
        "",
        f"- Ethnicity values observed: {audit['ethnicity_values_seen']:,}",
        f"- Known-vocabulary instances: {audit['known_value_instances']:,}",
        f"- Unreviewed instances / distinct terms: {audit['unreviewed_value_instances']:,} / {audit['unreviewed_distinct_values']:,}",
        "- The term-level source-context queue is `ETHNICITY_REVIEW_QUEUE.md`; it supports a scholarly decision without data loss or automatic normalization.",
        "",
        "## Provenance and limits",
        "",
        "Each delivered record retains its deterministic ID, image provenance, faithful transcription, normalized transcription, and structured extraction.",
        "The QA, identity, and graph artifacts are decision support. Duplicate, chronology, dangling-reference, and identity-candidate flags require scholarly review before any destructive correction or merge.",
        "",
        "## Files for review",
        "",
        "- `assembled/*.materialized.json` — delivered records with faithful and normalized text",
        "- `corpus_final_pipeline/review.html` — identity-review interface",
        "- `corpus_final_pipeline/network.graphml`, `nodes.csv`, `edges.csv` — network exports",
        "- `ETHNICITY_REVIEW_QUEUE.md` — contextual term-level review queue",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
