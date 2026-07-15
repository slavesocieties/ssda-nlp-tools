"""Offline tests for the segmentation gold sheets + scorer."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ssda_nlp_tools.seggold import build_sheet, score_corrections


def test_build_sheet_embeds_valid_payload_and_escapes(tmp_path):
    vol = os.path.join(ROOT, "Text data/SSDA_0013_0023_Gemini_V2.json")
    html = str(tmp_path / "s.html")
    pred = str(tmp_path / "s.pred.json")
    info = build_sheet(vol, html, pred, max_pages=4, vol_id="T")
    assert info["entries"] > 0
    text = open(html, encoding="utf-8").read()
    m = re.search(r"const PAGES = (\[.*?\]), VOL =", text, re.S)
    pages = json.loads(m.group(1))
    assert sum(len(p["entries"]) for p in pages) == info["entries"]
    assert all("id" in e and "text" in e and "partial" in e
               for p in pages for e in p["entries"])
    assert "Download corrections.json" in text and "localStorage" in text
    # data cannot terminate the script tag
    assert "\\u003c" in text or "<" not in "".join(
        e["text"] for p in pages for e in p["entries"])


def test_scorer_computes_precision_and_error_taxonomy():
    pred = {"volume": "V", "pages": [
        {"image": "V-01.jpg", "entries": [
            {"id": "V-01-01", "text": "a", "partial": False},
            {"id": "V-01-02", "text": "b", "partial": False},
            {"id": "V-01-03", "text": "c", "partial": False}]},
        {"image": "V-02.jpg", "entries": [
            {"id": "V-02-01", "text": "d", "partial": False}]},
    ]}
    corrections = {"corrections": {
        "V-01-01": {"verdict": "correct"},
        "V-01-02": {"verdict": "merge"},     # spurious boundary
        "V-01-03": {"verdict": "split"},     # hides 2 real entries
        "V-02-01": {"verdict": "bad"},       # wrong
        "__miss_V-02.jpg": {"missing_start": True},   # a real entry missed
    }}
    s = score_corrections(pred, corrections)
    assert s["predicted_entries"] == 4 and s["judged"] == 4
    assert s["exact_correct"] == 1
    assert s["precision"] == 0.25                     # 1 of 4 exactly correct
    assert s["over_splits_merge"] == 1 and s["under_splits"] == 1 and s["wrong"] == 1
    assert s["false_positive_boundaries"] == 2        # merge + bad
    assert s["pages_with_missing_starts"] == 1
    assert s["recall_approx"] is not None


def test_scorer_perfect_review():
    pred = {"volume": "V", "pages": [{"image": "V-01.jpg", "entries": [
        {"id": "V-01-01", "text": "a", "partial": False},
        {"id": "V-01-02", "text": "b", "partial": False}]}]}
    corrections = {"corrections": {"V-01-01": {"verdict": "correct"},
                                   "V-01-02": {"verdict": "correct"}}}
    s = score_corrections(pred, corrections)
    assert s["precision"] == 1.0 and s["recall_approx"] == 1.0
