import json

import pytest

from refresh_qa_labels import refresh_pipeline


def test_refreshes_json_and_human_summary_labels(tmp_path):
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    source = tmp_path / "701157.materialized.json"
    source.write_text(json.dumps({"volume": "701157", "entries": []}),
                      encoding="utf-8")
    (pipeline / "qa_report.json").write_text(
        json.dumps([{"volume": None, "issues": []}]), encoding="utf-8")
    (pipeline / "summary.txt").write_text(
        "Volume QA — None  (0 entries)\n", encoding="utf-8")
    assert refresh_pipeline(pipeline, [source]) == 1
    report = json.loads((pipeline / "qa_report.json").read_text(encoding="utf-8"))
    assert report[0]["volume"] == "701157"
    assert "Volume QA — 701157" in (pipeline / "summary.txt").read_text(encoding="utf-8")


def test_refuses_to_guess_when_summary_shape_disagrees(tmp_path):
    pipeline = tmp_path / "pipeline"; pipeline.mkdir()
    source = tmp_path / "701157.materialized.json"
    source.write_text(json.dumps({"volume": "701157"}), encoding="utf-8")
    (pipeline / "qa_report.json").write_text(
        json.dumps([{"volume": None}]), encoding="utf-8")
    (pipeline / "summary.txt").write_text("no QA header", encoding="utf-8")
    with pytest.raises(ValueError, match="unlabeled QA headers"):
        refresh_pipeline(pipeline, [source])
