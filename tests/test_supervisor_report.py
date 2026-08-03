from collections import Counter

import pytest

from build_supervisor_report import (
    corpus_graph_stats,
    delivery_accounting,
    graph_validation_counts,
    ledger_accounting,
    render_report,
)


def _summary():
    return {
        "volumes": {
            "1": {"corpus_records": 10, "materialized_records": 8,
                  "missing_records": 0, "invalid_batches": 0,
                  "partials_dropped": 1, "state": "COMPLETE"},
        },
        "totals": {"corpus_records": 10, "materialized_records": 8,
                   "missing_records": 0, "invalid_batches": 0,
                   "withdrawn_records": 1},
    }


def _ledger(reserved=0.0, active=False):
    jobs = []
    if active:
        jobs.append({"job_id": "batch_1", "status": "submitted", "volume": "1",
                     "reserved_usd": reserved})
    return {"cap_usd": 35, "confirmed_usd": 10, "reserved_usd": reserved,
            "jobs": jobs}


def _audit():
    return {"ethnicity_values_seen": 3, "known_value_instances": 2,
            "unreviewed_value_instances": 1, "unreviewed_distinct_values": 1}


def _graph():
    return {"mentions": 9, "identities": 8, "cross_chunk_identities": 1,
            "network_nodes": 8, "network_edges": 4}


def test_report_derives_withdrawals_partials_and_active_reservation():
    report = render_report(_summary(), _ledger(0.08, active=True), _audit(),
                           Counter({"duplicate_entry": 2}), _graph())
    assert "Page-truncated source records" in report and ": 1" in report
    assert "Withdrawn records" in report and ": 1" in report
    assert "Current reservations: $0.080000" in report
    assert "Outstanding paid work: 1 submitted job" in report
    assert "No reservation remains" not in report
    assert "26.6%" not in report


def test_report_says_no_recorded_job_only_when_balance_is_zero():
    report = render_report(_summary(), _ledger(), _audit(), Counter(), _graph())
    assert "No submitted provider job with a recorded job ID" in report


def test_delivery_accounting_rejects_mismatched_totals():
    summary = _summary()
    summary["totals"]["materialized_records"] = 9
    with pytest.raises(ValueError, match="materialized_records"):
        delivery_accounting(summary)


def test_ledger_accounting_rejects_untracked_reservation():
    with pytest.raises(ValueError, match="active-job reservations"):
        ledger_accounting(_ledger(0.08, active=False))


def test_graph_stats_use_machine_json_and_fail_on_missing_fields(tmp_path):
    (tmp_path / "network.json").write_text(
        '{"nodes":[{"id":"p1"}],"edges":[],"stats":{"nodes":1,"edges":0}}',
        encoding="utf-8")
    (tmp_path / "person_index.json").write_text(
        '[{"n_mentions":2,"cross_chunk":false}]', encoding="utf-8")
    assert corpus_graph_stats(tmp_path) == {
        "mentions": 2, "identities": 1, "cross_chunk_identities": 0,
        "network_nodes": 1, "network_edges": 0}
    (tmp_path / "network.json").write_text(
        '{"nodes":[],"edges":[],"stats":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="required machine fields"):
        corpus_graph_stats(tmp_path)


def test_graph_validation_deduplicates_repeated_pair_hits(tmp_path):
    path = tmp_path / "graph_validation.json"
    path.write_text(
        '{"missing_inverse":[],"contradictory_roles":['
        '{"a":"p1","b":"p2"},{"a":"p1","b":"p2"},'
        '{"a":"p2","b":"p1"}],"ancestry_cycle":['
        '{"from":"p1","to":"p2"}]}', encoding="utf-8")
    counts = graph_validation_counts(path)
    assert counts["contradictory_raw"] == 3
    assert counts["contradictory_directed_pairs"] == 2
    assert counts["contradictory_unordered_pairs"] == 1
