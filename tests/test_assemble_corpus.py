"""Tests for the offline post-batch corpus assembly (assemble_corpus.py)."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module():
    path = os.path.join(ROOT, "assemble_corpus.py")
    spec = importlib.util.spec_from_file_location("assemble_corpus", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_volume_of_handles_plain_and_aliased_custom_ids():
    v = _module()._volume_of
    assert v("176899-b0000") == "176899"
    assert v("luna-production-701054-b0004") == "701054"      # historical alias prefix
    assert v("29597-b0012") == "29597"
    assert v("nothing-here") is None


def _resp_row(custom_id, content, status=200, finish="stop"):
    return {"custom_id": custom_id, "response": {"status_code": status,
            "body": {"choices": [{"finish_reason": finish,
                                  "message": {"content": content}}]}}}


def test_read_rows_groups_by_volume_and_separates_invalid(tmp_path):
    mod = _module()
    good = json.dumps({"results": [
        {"entry": "701054-0001-01", "normalized": "x", "data": {"people": [], "events": []}}]})
    # one valid 701054 row, one 176899 row that errored (non-200), one non-stop
    rows = [
        _resp_row("701054-b0000", good),
        _resp_row("176899-b0000", good, status=500),
        _resp_row("176899-b0001", good, finish="length"),
    ]
    (tmp_path / "j.output.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    by = mod.read_rows_by_volume(tmp_path)
    assert set(by["701054"]["valid"]) == {"701054-0001-01"}    # valid row parsed
    assert by["701054"]["invalid"] == []
    assert by["176899"]["valid"] == {}                         # both 176899 rows rejected
    assert len(by["176899"]["invalid"]) == 2                   # 500 + non-stop, flagged not dropped
