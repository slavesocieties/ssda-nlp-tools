#!/usr/bin/env python3
"""Build a factual, offline supervisor handoff from validated artifacts.

The report fails closed on inconsistent delivery accounting, ledger balances,
or graph artifacts. It never carries forward prose claims from an older run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def delivery_accounting(summary: dict) -> dict:
    volumes = summary.get("volumes")
    totals = summary.get("totals")
    if not isinstance(volumes, dict) or not volumes or not isinstance(totals, dict):
        raise ValueError("CORPUS_SUMMARY must contain non-empty volumes and totals")
    fields = ("corpus_records", "materialized_records", "missing_records",
              "invalid_batches")
    sums = {}
    for field in fields:
        try:
            sums[field] = sum(int(value[field]) for value in volumes.values())
            reported = int(totals[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing/non-numeric delivery field: {field}") from exc
        if sums[field] != reported:
            raise ValueError(
                f"delivery accounting mismatch for {field}: volume sum "
                f"{sums[field]} != total {reported}")
    partials = sum(int(value.get("partials_dropped", 0))
                   for value in volumes.values())
    withdrawals = int(totals.get("withdrawn_records", 0))
    rejected = sum(int(value.get("rejected_requests", 0))
                   for value in volumes.values())
    if "rejected_requests" in totals and rejected != int(totals["rejected_requests"]):
        raise ValueError(
            f"delivery accounting mismatch for rejected_requests: volume sum "
            f"{rejected} != total {totals['rejected_requests']}")
    expected_delivered = (sums["corpus_records"] - sums["missing_records"]
                          - partials - withdrawals)
    if sums["materialized_records"] != expected_delivered:
        raise ValueError(
            "delivery accounting mismatch: delivered must equal source minus "
            "missing, dropped partials, and withdrawals")
    return {**sums, "partials_dropped": partials, "rejected_requests": rejected,
            "withdrawn_records": withdrawals}


def ledger_accounting(ledger: dict) -> dict:
    try:
        cap = float(ledger["cap_usd"])
        confirmed = float(ledger["confirmed_usd"])
        reserved = float(ledger["reserved_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ledger is missing numeric cap/confirmed/reserved fields") from exc
    if min(cap, confirmed, reserved) < 0 or confirmed + reserved > cap + 1e-9:
        raise ValueError("ledger balances are negative or exceed the approved cap")
    jobs = ledger.get("jobs") or []
    active = [job for job in jobs
              if job.get("status") == "submitted" and job.get("job_id")]
    active_reserved = sum(float(job.get("reserved_usd", 0)) for job in active)
    if abs(active_reserved - reserved) > 1e-7:
        raise ValueError(
            f"ledger reserved balance ${reserved:.7f} does not equal active-job "
            f"reservations ${active_reserved:.7f}")
    unresolved_markers = [job for job in jobs
                          if job.get("status") in {"submitted", "submitted_pending"}
                          and not job.get("job_id")]
    return {"cap_usd": cap, "confirmed_usd": confirmed,
            "reserved_usd": reserved, "headroom_usd": cap - confirmed - reserved,
            "active_jobs": active, "unresolved_markers": unresolved_markers}


def qa_counts(live: Path, volumes: list[str]) -> Counter:
    result = Counter()
    for vol in volumes:
        data = read_json(live / f"{vol}_final_pipeline" / "qa_report.json")
        for report in data if isinstance(data, list) else [data]:
            for issue in report.get("issues") or []:
                result[issue.get("type", "unknown")] += 1
    return result


def corpus_graph_stats(graph_dir: Path) -> dict:
    network = read_json(graph_dir / "network.json")
    people = read_json(graph_dir / "person_index.json")
    if not isinstance(network, dict) or not isinstance(people, list):
        raise ValueError("graph artifacts have unexpected top-level types")
    nodes = network.get("nodes")
    edges = network.get("edges")
    stats = network.get("stats")
    if not isinstance(nodes, list) or not isinstance(edges, list) \
            or not isinstance(stats, dict):
        raise ValueError("network.json is missing nodes, edges, or stats")
    try:
        mentions = sum(int(person["n_mentions"]) for person in people)
        cross_chunk = sum(bool(person["cross_chunk"]) for person in people)
        stat_nodes = int(stats["nodes"])
        stat_edges = int(stats["edges"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("graph artifacts are missing required machine fields") from exc
    if stat_nodes != len(nodes) or stat_nodes != len(people):
        raise ValueError("graph node/identity counts disagree across artifacts")
    if stat_edges != len(edges):
        raise ValueError("graph edge count disagrees with network edge rows")
    return {"mentions": mentions, "identities": len(people),
            "cross_chunk_identities": cross_chunk,
            "network_nodes": stat_nodes, "network_edges": stat_edges}


def prompt_metric_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    data = read_json(path)
    title = data.get("title", "Measured prompt result") if isinstance(data, dict) else None
    statements = data.get("statements") if isinstance(data, dict) else None
    if not isinstance(title, str) or not title.strip() or not isinstance(statements, list) \
            or not statements or any(not isinstance(item, str) or not item.strip()
                                      for item in statements):
        raise ValueError("prompt metrics must contain a title and non-empty statements")
    return [f"## {title.strip()}", "", *[item.strip() for item in statements], ""]


def graph_validation_counts(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("graph validation artifact must be a JSON object")
    required = ("missing_inverse", "contradictory_roles", "ancestry_cycle")
    if any(not isinstance(data.get(field), list) for field in required):
        raise ValueError("graph validation artifact is missing required finding lists")
    missing_inverse = {
        (str(row.get("source")), str(row.get("target")), str(row.get("type")),
         str(row.get("expected"))) for row in data["missing_inverse"]}
    directed_roles = {(str(row.get("a")), str(row.get("b")))
                      for row in data["contradictory_roles"]}
    unordered_roles = {tuple(sorted(pair)) for pair in directed_roles}
    cycle_edges = {(str(row.get("from")), str(row.get("to")))
                   for row in data["ancestry_cycle"]}
    return {"missing_inverse": len(missing_inverse),
            "contradictory_raw": len(data["contradictory_roles"]),
            "contradictory_directed_pairs": len(directed_roles),
            "contradictory_unordered_pairs": len(unordered_roles),
            "ancestry_cycle_edges": len(cycle_edges)}


def render_report(summary: dict, ledger: dict, audit: dict, qa: Counter,
                  graph: dict, metrics: list[str] | None = None,
                  graph_validation: dict | None = None) -> str:
    delivery = delivery_accounting(summary)
    spend = ledger_accounting(ledger)
    volumes = summary["volumes"]
    validation_complete = (delivery["missing_records"] == 0
                           and delivery["invalid_batches"] == 0)
    lines = [
        "# SSDA Corpus Extraction: Supervisor Results",
        "",
        "## Delivery status",
        "",
        f"- Source records: {delivery['corpus_records']:,}",
        f"- Delivered records: {delivery['materialized_records']:,}",
        f"- Missing source records: {delivery['missing_records']:,}",
        f"- Invalid provider batches represented in delivery: {delivery['invalid_batches']:,}",
        f"- Historically rejected provider requests retained in the audit trail: {delivery['rejected_requests']:,}",
        f"- Page-truncated source records retained for audit and excluded from delivery: {delivery['partials_dropped']:,}",
        f"- Withdrawn records retained in quarantine and excluded from delivery: {delivery['withdrawn_records']:,}",
        "",
        "| Volume | Delivered | Source records | Partials excluded | Delivery state |",
        "|---|---:|---:|---:|---|",
    ]
    for vol, value in volumes.items():
        lines.append(
            f"| {vol} | {int(value['materialized_records']):,} | "
            f"{int(value['corpus_records']):,} | "
            f"{int(value.get('partials_dropped', 0)):,} | {value['state']} |")
    lines += [
        "",
        "## Cost and delivery controls",
        "",
        f"- Confirmed OpenAI Batch spend: ${spend['confirmed_usd']:.6f}",
        f"- Current reservations: ${spend['reserved_usd']:.6f}",
        f"- Approved cumulative cap: ${spend['cap_usd']:.2f}",
        f"- Remaining unreserved headroom: ${spend['headroom_usd']:.6f}",
    ]
    if spend["active_jobs"]:
        volumes_active = sorted({str(job.get("volume") or "unknown")
                                 for job in spend["active_jobs"]})
        lines.append(
            f"- Outstanding paid work: {len(spend['active_jobs'])} submitted job(s) "
            f"for volume(s) {', '.join(volumes_active)}; their reservations are "
            "included above.")
    else:
        lines.append("- No submitted provider job with a recorded job ID remains outstanding.")
    if spend["unresolved_markers"]:
        lines.append(
            f"- Ledger audit note: {len(spend['unresolved_markers'])} legacy pending "
            "marker(s) lack a provider job ID and are reported separately from "
            "active reservations.")
    if validation_complete:
        lines.append(
            "- The assembled delivery summary reports zero missing records and zero "
            "invalid batches; assembly consumed only request-level accepted artifacts.")
    else:
        lines.append(
            "- Delivery is not yet final: missing records or invalid batches remain "
            "visible in the accounting above.")
    lines += [
        "- Raw provider outputs are retained for audit but cannot feed delivery directly.",
        "",
    ]
    lines += metrics or []
    lines += [
        "## QA and graph outputs",
        "",
        f"- Person mentions: {graph['mentions']:,}",
        f"- Resolved identities: {graph['identities']:,}",
        f"- Identities spanning more than one source volume: {graph['cross_chunk_identities']:,}",
        f"- Network nodes / typed edges: {graph['network_nodes']:,} / {graph['network_edges']:,}",
        "- The browser review interface is a capped, ranked decision aid; this report "
        "does not claim that its displayed rows are the full candidate universe.",
        "",
        "QA flags are preserved for review; they do not delete or rewrite records:",
        "",
        "| QA flag | Count |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {count:,} |" for name, count in qa.most_common()]
    if graph_validation is not None:
        clean = not any((graph_validation["missing_inverse"],
                         graph_validation["contradictory_unordered_pairs"],
                         graph_validation["ancestry_cycle_edges"]))
        lines += ["", "### Graph invariant review", ""]
        if clean:
            lines.append("- All recorded graph invariants passed.")
        else:
            lines += [
                f"- Missing reciprocal relationship edges: {graph_validation['missing_inverse']:,}",
                f"- Unique person pairs carrying contradictory relationship roles: {graph_validation['contradictory_unordered_pairs']:,} "
                f"({graph_validation['contradictory_raw']:,} raw validator hits)",
                f"- Ancestry-cycle back edges requiring review: {graph_validation['ancestry_cycle_edges']:,}",
                "- These findings remain unresolved review items. They do not alter "
                "the extracted records, but the graph must not be described as "
                "logically clean until they are adjudicated.",
            ]
    lines += [
        "",
        "## Ethnicity review queue",
        "",
        f"- Ethnicity values observed: {int(audit['ethnicity_values_seen']):,}",
        f"- Known-vocabulary instances: {int(audit['known_value_instances']):,}",
        f"- Unreviewed instances / distinct terms: {int(audit['unreviewed_value_instances']):,} / {int(audit['unreviewed_distinct_values']):,}",
        "- The term-level source-context queue is `ETHNICITY_REVIEW_QUEUE.md`; it "
        "supports scholarly review without data loss or automatic normalization.",
        "",
        "## Provenance and limits",
        "",
        "Each delivered record retains its deterministic ID, image provenance, faithful "
        "transcription, normalized transcription, and structured extraction.",
        "QA, identity, and graph artifacts are decision support. Duplicate, chronology, "
        "dangling-reference, and identity-candidate flags require scholarly review before "
        "any destructive correction or merge.",
        "",
        "## Files for review",
        "",
        "- `assembled/*.materialized.json` - delivered records with faithful and normalized text",
        "- `corpus_final_pipeline/review.html` - capped identity-review interface",
        "- `corpus_final_pipeline/network.graphml`, `nodes.csv`, `edges.csv` - network exports",
        "- `graph_validation.json` - unresolved logical-invariant review findings",
        "- `ETHNICITY_REVIEW_QUEUE.md` - contextual term-level review queue",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", type=Path, default=Path("production/luna_v3"))
    ap.add_argument("--ledger", type=Path,
                    default=Path("production/luna_live/spend_ledger.json"))
    ap.add_argument("--audit", type=Path,
                    default=Path("production/luna_v3/ETHNICITY_AUDIT.json"))
    ap.add_argument("--graph-dir", type=Path,
                    help="machine graph-artifact directory (default: LIVE/corpus_final_pipeline)")
    ap.add_argument("--prompt-metrics", type=Path,
                    help="optional measured JSON with title and statements; omitted by default")
    ap.add_argument("--graph-validation", type=Path,
                    help="optional validate_graph.py JSON; unresolved findings are "
                         "reported, never hidden")
    ap.add_argument("--out", type=Path,
                    default=Path("production/luna_v3/SUPERVISOR_RESULTS.md"))
    args = ap.parse_args(argv)

    summary = read_json(args.live / "CORPUS_SUMMARY.json")
    ledger = read_json(args.ledger)
    audit = read_json(args.audit)
    volumes = list(summary.get("volumes") or {})
    qa = qa_counts(args.live, volumes)
    graph = corpus_graph_stats(args.graph_dir or args.live / "corpus_final_pipeline")
    report = render_report(
        summary, ledger, audit, qa, graph, prompt_metric_lines(args.prompt_metrics),
        graph_validation_counts(args.graph_validation))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"REFUSING: {exc}")
