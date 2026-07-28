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
