import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("materialize", os.path.join(ROOT, "materialize_luna_results.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_materialize_keeps_faithful_text_and_partial_flag():
    corpus = {"volume": "1", "entries": [{"id": "1-01", "text": "faithful", "images": ["a.jpg"], "partial": True}]}
    extracted = {"1-01": {"normalized": "normal", "data": {"people": [], "events": []}}}
    out = m.materialize(corpus, extracted)
    assert out["entries"][0]["text_faithful"] == "faithful"
    assert out["entries"][0]["normalized"] == "normal"
    assert out["entries"][0]["partial"] is True


def test_materialize_partial_is_explicitly_marked():
    corpus = {"volume": "1", "entries": [{"id": "1-01", "text": "a"}, {"id": "1-02", "text": "b"}]}
    extracted = {"1-01": {"normalized": "a", "data": {"people": [], "events": []}}}
    out = m.materialize(corpus, extracted, allow_incomplete=True)
    assert len(out["entries"]) == 1
    assert out["coverage"] == {"corpus_records": 2, "materialized_records": 1,
                               "missing_records": 1, "incomplete": True}
