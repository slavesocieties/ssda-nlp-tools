"""Security regressions: the review pages embed model-generated text.

Every review page interpolates text an LLM produced from an archival image into
a file a human then opens in a browser. That is a stored-injection shape, and the
mitigation is currently correct but was pinned by nothing: all three renderers
embed their data as JSON inside a <script> block and escape "<" to \\u003c by
hand. `json.dumps` does NOT escape "/" or "<", so dropping that one `.replace`
during a refactor silently reopens a </script> breakout, and the page would look
completely normal in review.

These tests exist so that deletion fails loudly.

The audit that produced them (2026-08-05) found no credential material in the
working tree or in all 140 commits of history, no eval/exec/shell=True, and
secrets read from the environment only -- with no CLI flag anywhere that could
put a key into shell history.
"""
import json
import os
import re
import subprocess
import tempfile

import pytest

BREAKOUT = "Maria</script><script>window.PWNED=1</script>"
ESCAPED_MARKER = "window.PWNED=1</script>"


def _pair(name):
    return [{"score": 0.9, "reasons": ["name~1.00"], "weight": 1, "stratum": "s",
             "a": {"entry": "E1", "id": "P01", "name": name, "detail": {}},
             "b": {"entry": "E2", "id": "P02", "name": "Ana", "detail": {}}}]


def _render(fn, arg):
    out = os.path.join(tempfile.mkdtemp(), "p.html")
    fn(arg, out)
    return open(out, encoding="utf-8").read()


def test_pairwise_review_blocks_script_breakout():
    from ssda_nlp_tools.review_html import render_review_html
    page = _render(render_review_html, _pair(BREAKOUT))
    assert ESCAPED_MARKER not in page
    assert "u003c" in page, "the < escape is gone; a breakout is now possible"


def test_likelihood_review_blocks_script_breakout():
    from ssda_nlp_tools.likelihood_review_html import render_likelihood_review_html
    page = _render(render_likelihood_review_html, _pair(BREAKOUT))
    assert ESCAPED_MARKER not in page
    assert "u003c" in page


def test_person_review_blocks_script_breakout():
    from ssda_nlp_tools.person_review_html import render_person_review_html
    rows = [{"person": {"entry": "E1", "id": "P01", "name": BREAKOUT, "detail": {}},
             "candidates": [{"score": 0.9, "reasons": ["x"],
                             "other": {"entry": "E2", "id": "P02",
                                       "name": "Ana", "detail": {}}}]}]
    page = _render(render_person_review_html, rows)
    assert ESCAPED_MARKER not in page
    assert "u003c" in page


def test_relationship_review_escapes_html():
    """This one renders into markup rather than JSON, so it escapes instead."""
    from run_relationship_review import collect, render
    entries = {"E1": {"id": "E1", "text_faithful": BREAKOUT,
                      "data": {"people": []}}}
    buckets = {"dangling_relationship": [{"entry": "E1", "detail": BREAKOUT}]}
    page = render(collect(entries, buckets, 1), {}, 1)
    assert ESCAPED_MARKER not in page
    assert "&lt;script&gt;" in page


def test_no_source_file_carries_credential_material():
    """Pattern-based, never the literal secret: putting a real key in a test is
    the leak this is meant to catch."""
    pats = [re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*=\s*[\"'][^\"'\s]{12,}")]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", "production",
                                    ".pytest_cache", ".venv", "node_modules"}]
        for f in files:
            if not f.endswith((".py", ".md", ".json", ".txt", ".yml", ".yaml")):
                continue
            p = os.path.join(dirpath, f)
            try:
                if os.path.getsize(p) > 4_000_000:
                    continue
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for rx in pats:
                if rx.search(text):
                    bad.append(os.path.relpath(p, root))
    assert not bad, f"credential-shaped material in: {sorted(set(bad))[:5]}"


def test_no_cli_flag_accepts_a_secret():
    """A secret passed as an argument lands in shell history and in ps output.
    build_adjudication_set.py documents having deliberately no --api-key flag."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rx = re.compile(r"add_argument\(\s*[\"']--(?:[a-z-]*-)?"
                    r"(api[-_]?key|secret|token|password)\b")
    bad = []
    for f in os.listdir(root):
        if f.endswith(".py"):
            text = open(os.path.join(root, f), encoding="utf-8",
                        errors="ignore").read()
            if rx.search(text):
                bad.append(f)
    assert not bad, f"these accept a secret on the command line: {bad}"
