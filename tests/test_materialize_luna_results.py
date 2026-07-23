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
