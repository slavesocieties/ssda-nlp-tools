"""The delivered-label lock must be able to FAIL, or it is decoration.

Every label set in this project is keyed by POSITION and carries no pair
identifiers, so a returned grade is only "row 41 = yes". Point that grade at a
regenerated file and it silently describes a different pair. The repo has hit
this twice; on 2026-08-12 it stopped being hypothetical, because
`blocked_pairs.html` went to Daniel on Slack and its grades are outstanding.

`self_check.py` locks the delivered artifacts by sha256. These tests exist
because a checker that cannot fail is worse than no checker -- it converts an
unverified claim into a verified-looking one. So they tamper on purpose.
"""
import json
import os
import shutil

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, "delivered_labels.lock.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(LOCK),
    reason="no delivered_labels.lock.json in this checkout")


def _check():
    """Run just the lock check, out of self_check, against the real tree."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_sc", os.path.join(ROOT, "self_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, fn in mod.CHECKS:
        if "byte-identical" in name:
            return fn(ROOT)
    raise AssertionError("the delivered-label check is gone from self_check")


def _locked_paths():
    lock = json.load(open(LOCK, encoding="utf-8"))
    return [os.path.join(ROOT, p) for p in lock["entries"]]


def test_lock_passes_on_the_real_tree():
    ok, detail = _check()
    assert ok is True, detail


def test_every_locked_artifact_actually_exists():
    for p in _locked_paths():
        assert os.path.exists(p), f"locked artifact missing: {p}"


def test_a_single_appended_byte_is_caught(tmp_path):
    """The realistic regeneration is a rewrite, but one byte must trip it."""
    target = _locked_paths()[0]
    backup = str(tmp_path / "backup")
    shutil.copy2(target, backup)
    try:
        with open(target, "ab") as fh:
            fh.write(b"<!-- regenerated -->")
        ok, detail = _check()
        assert ok is False, "a modified delivered artifact passed the lock"
        assert "no longer line up" in detail
    finally:
        shutil.copy2(backup, target)
    assert _check()[0] is True, "restore did not return the tree to locked state"


def test_a_deleted_artifact_is_caught_too(tmp_path):
    """Deletion must not read as 'nothing changed'."""
    target = _locked_paths()[0]
    backup = str(tmp_path / "backup")
    shutil.copy2(target, backup)
    try:
        os.remove(target)
        ok, detail = _check()
        assert ok is False
        assert "GONE" in detail
    finally:
        shutil.copy2(backup, target)
    assert _check()[0] is True


def test_the_blocked_sample_still_carries_its_weights():
    """The property the recall estimate depends on, not just the bytes."""
    p = os.path.join(ROOT, "production/luna_v3/blocked_labels/blocked_pairs.json")
    if not os.path.exists(p):
        pytest.skip("blocked sample not present in this checkout")
    d = json.load(open(p, encoding="utf-8"))
    assert len(d["pairs"]) == 200
    assert d["population"] == 3013215
    # Horvitz-Thompson: the weights must sum to the population they stand for,
    # or sum(weight * grade) is not an estimate of anything.
    assert abs(sum(r["weight"] for r in d["pairs"]) - d["population"]) < 1.0
