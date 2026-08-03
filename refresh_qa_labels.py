#!/usr/bin/env python3
"""Refresh missing volume labels in already-generated QA artifacts, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def refresh_pipeline(pipeline: Path, sources: list[Path]) -> int:
    report_path = pipeline / "qa_report.json"
    summary_path = pipeline / "summary.txt"
    reports = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(reports, list):
        reports = [reports]
    if len(reports) != len(sources):
        raise ValueError(
            f"{pipeline}: {len(reports)} QA reports but {len(sources)} sources")
    labels = []
    missing_labels = []
    for report, source_path in zip(reports, sources):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        label = str(source.get("volume") or source.get("title") or source.get("id") or "")
        if not label:
            raise ValueError(f"source has no volume label: {source_path}")
        labels.append(label)
        if not report.get("volume"):
            missing_labels.append(label)
            report["volume"] = label
    if not missing_labels:
        return 0
    text = summary_path.read_text(encoding="utf-8")
    marker = "Volume QA — None"
    if text.count(marker) != len(missing_labels):
        raise ValueError(
            f"{summary_path}: expected {len(missing_labels)} unlabeled QA headers, "
            f"found {text.count(marker)}")
    for label in missing_labels:
        text = text.replace(marker, f"Volume QA — {label}", 1)
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(text, encoding="utf-8")
    return len(missing_labels)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live", type=Path)
    args = parser.parse_args(argv)
    corpus = json.loads((args.live / "CORPUS_SUMMARY.json").read_text(encoding="utf-8"))
    volumes = list(corpus.get("volumes") or {})
    sources = [args.live / "assembled" / f"{volume}.materialized.json"
               for volume in volumes]
    changed = 0
    for volume, source in zip(volumes, sources):
        changed += refresh_pipeline(args.live / f"{volume}_final_pipeline", [source])
    changed += refresh_pipeline(args.live / "corpus_final_pipeline", sources)
    print(f"refreshed {changed} missing QA volume label(s) under {args.live}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"REFUSING: {exc}")
