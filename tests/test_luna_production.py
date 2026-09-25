import importlib.util
import json
import os
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("run_luna_production", os.path.join(ROOT, "run_luna_production.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _row(custom_id="1-b0000"):
    return {"custom_id": custom_id, "tail_message": {"role": "user", "content": json.dumps({"entries": [{"entry": "1-01"}]})}}


def _response(custom_id="1-b0000", entry="1-01", finish="stop"):
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"finish_reason": finish, "message": {
                    "content": json.dumps({"results": [{"entry": entry}]})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25},
            },
        },
    }


def test_validate_output_requires_exact_ids_and_normal_stop():
    valid = runner.validate_output([_row()], [_response()])
    assert valid["valid"]
    assert valid["confirmed_usd_conservative"] == 0.000125
    assert not runner.validate_output([_row()], [_response(entry="wrong")])["valid"]
    assert not runner.validate_output([_row()], [_response(finish="length")])["valid"]


def test_invalid_batch_salvages_only_individually_valid_requests(tmp_path):
    good = _response("1-b0000")
    extra = _response("1-b0001", entry="wrong")
    rows = [_row("1-b0000"), _row("1-b0001")]
    report = runner.validate_output(rows, [good, extra])
    assert not report["valid"]
    assert report["accepted_custom_ids"] == ["1-b0000"]
    assert set(report["rejected_custom_ids"]) == {"1-b0001"}
    out = tmp_path / "accepted.jsonl"
    runner.write_accepted_output(out, [good, extra], report["accepted_custom_ids"])
    assert [json.loads(line)["custom_id"] for line in out.read_text().splitlines()] == ["1-b0000"]


def test_validate_output_rejects_schema_invalid_data_even_when_ids_match():
    response = _response()
    response["response"]["body"]["choices"][0]["message"]["content"] = json.dumps({
        "results": [{"entry": "1-01", "normalized": "x",
                     "data": {"people": ["not-an-object"], "events": []}}]})
    report = runner.validate_output([_row()], [response])
    assert not report["valid"]
    assert report["accepted_custom_ids"] == []
    assert "invalid extraction schema" in report["rejected_custom_ids"]["1-b0000"]


def test_validate_output_rejects_duplicate_result_ids():
    response = _response()
    response["response"]["body"]["choices"][0]["message"]["content"] = json.dumps({
        "results": [{"entry": "1-01"}, {"entry": "1-01"}]})
    report = runner.validate_output([_row()], [response])
    assert not report["valid"]
    assert report["accepted_custom_ids"] == []


def test_historical_prefix_is_normalized_only_for_request_aliases():
    assert runner.normal_id("luna-production-701054-b0013") == "701054-b0013"
    assert runner.normal_id("701054-b0013") == "701054-b0013"


def test_reextract_run_id_namespaces_requests_without_changing_source_ids():
    assert runner.with_run_id("176899-b0000", "v2") == "v2-176899-b0000"
    assert runner.with_run_id("176899-b0000", "") == "176899-b0000"


def test_nondefault_output_requires_an_explicit_shared_ledger():
    try:
        runner.resolve_ledger_path(Path("production/luna_v2"), None)
    except ValueError as exc:
        assert "--ledger-path" in str(exc)
    else:
        raise AssertionError("separate output directory silently created a ledger")
    assert runner.resolve_ledger_path(
        Path("production/luna_v2"), Path("production/luna_live/spend_ledger.json")
    ) == Path("production/luna_live/spend_ledger.json")


def test_namespaced_reextract_requires_isolated_artifact_directory():
    try:
        runner.require_isolated_output_for_run_id(Path("production/luna_live"), "v3")
    except ValueError as exc:
        assert "--outdir" in str(exc)
    else:
        raise AssertionError("namespaced re-extraction can overwrite live artifacts")
    runner.require_isolated_output_for_run_id(Path("production/luna_v3"), "v3")


def _ledger(tmp_path, cap):
    path = tmp_path / "spend_ledger.json"
    path.write_text(json.dumps({"cap_usd": cap, "confirmed_usd": 11.0,
                                "reserved_usd": 1.0, "jobs": []}), encoding="utf-8")
    return path


def test_default_cap_is_200_and_new_ledgers_record_it(tmp_path):
    assert runner.DEFAULT_CAP_USD == 200.0
    ledger = runner.load_ledger(tmp_path / "new.json", runner.DEFAULT_CAP_USD)
    assert ledger["cap_usd"] == 200.0


def test_existing_ledger_cap_is_only_raised_deliberately(tmp_path):
    path = _ledger(tmp_path, 20.0)
    try:
        runner.load_ledger(path, 200.0)
        raise AssertionError("a changed cap must not be accepted silently")
    except ValueError as exc:
        assert "--raise-cap" in str(exc)
    raised = runner.load_ledger(path, 200.0, raise_cap=True)
    assert raised["cap_usd"] == 200.0
    assert raised["cap_history"] == [{"from_usd": 20.0, "to_usd": 200.0}]
    assert raised["confirmed_usd"] == 11.0          # committed spend is untouched


def test_recorded_cap_is_never_lowered(tmp_path):
    path = _ledger(tmp_path, 200.0)
    try:
        runner.load_ledger(path, 20.0, raise_cap=True)
        raise AssertionError("lowering the cap must be refused")
    except ValueError as exc:
        assert "never lowered" in str(exc)


def test_raise_cap_dry_run_leaves_ledger_unchanged_and_confirm_writes_it(tmp_path, capsys):
    path = _ledger(tmp_path, 20.0)
    batch = tmp_path / "1.batches.jsonl"
    batch.write_text(json.dumps({"header": {"volume": "1", "prefix_messages": []}}) + "\n"
                     + json.dumps(_row()) + "\n", encoding="utf-8")
    args = [str(batch), "--outdir", str(tmp_path), "--ledger-path", str(path), "--raise-cap"]
    assert runner.main(args) == 0
    assert "would raise ledger cap $20 -> $200" in capsys.readouterr().out
    assert json.loads(path.read_text(encoding="utf-8"))["cap_usd"] == 20.0
    # --confirm with nothing left to submit: the raise is recorded, nothing is sent
    ledger = json.loads(path.read_text(encoding="utf-8"))
    ledger["jobs"] = [{"status": "validated", "custom_id": "1-b0000"}]
    path.write_text(json.dumps(ledger), encoding="utf-8")
    assert runner.main(args + ["--confirm"]) == 0
    out = capsys.readouterr().out
    assert "RAISED ledger cap $20 -> $200" in out and "No unsent compact requests" in out
    assert json.loads(path.read_text(encoding="utf-8"))["cap_usd"] == 200.0
