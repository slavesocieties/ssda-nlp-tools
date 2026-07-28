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
