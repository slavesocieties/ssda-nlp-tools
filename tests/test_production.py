"""Tests for the deterministic production build (run_production.py)."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _production_module():
    path = os.path.join(ROOT, "run_production.py")
    spec = importlib.util.spec_from_file_location("run_production", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disposition_production_requires_all_pages_deterministic():
    d = _production_module().disposition
    routes = {"a.jpg": "deterministic-sacramental", "b.jpg": "deterministic-sacramental"}
    assert d(["a.jpg", "b.jpg"], routes) == "production"


def test_disposition_worst_page_wins():
    """A record spanning a deterministic page AND a fallback/error page is NOT
    production — it inherits the harder path so nothing paid or broken leaks
    into the free output."""
    d = _production_module().disposition
    base = {"a.jpg": "deterministic-sacramental"}
    assert d(["a.jpg", "b.jpg"], {**base, "b.jpg": "luna-sacramental-fallback"}) == "needs-fallback"
    assert d(["a.jpg", "b.jpg"], {**base, "b.jpg": "retranscribe"}) == "blocked-retranscribe"
    assert d(["a.jpg", "b.jpg"], {**base, "b.jpg": "skip-index"}) == "index-context"
    # re-transcribe is the hardest: a broken source page blocks the record even
    # if a fallback page is also present
    assert d(["a.jpg", "b.jpg", "c.jpg"],
             {"a.jpg": "deterministic-sacramental", "b.jpg": "luna-sacramental-fallback",
              "c.jpg": "retranscribe"}) == "blocked-retranscribe"


def test_disposition_unknown_page_routes_to_review():
    d = _production_module().disposition
    assert d(["x.jpg"], {}) == "review"                       # page not in the manifest
    assert d([], {}) == "production"                          # no pages -> vacuously all-deterministic
