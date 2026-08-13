"""The recall estimator must refuse a moved sample and must not invent coverage.

Two failure modes, both of which would produce a confident wrong number:

1. Grades are keyed by ROW POSITION with no pair identifiers. Point them at a
   regenerated sample and every grade describes a different pair, while the
   output looks entirely normal. The estimator verifies the sample against
   delivered_labels.lock.json before reading a grade.

2. A stratum nobody graded is not a stratum of zeros. Treating the two the same
   manufactures "no merges hidden here" out of an absence of evidence, which is
   this repo's signature way of producing a clean wrong answer.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "estimate_blocked_recall.py")
SAMPLE = os.path.join(ROOT, "production/luna_v3/blocked_labels/blocked_pairs.json")

pytestmark = pytest.mark.skipif(not os.path.exists(SAMPLE),
                                reason="blocked sample not in this checkout")


def _run(grades_path, *extra):
    return subprocess.run(
        [sys.executable, "-X", "utf8", SCRIPT, "--grades", grades_path,
         "--root", ROOT, "--sample", SAMPLE, *extra],
        capture_output=True, text=True, cwd=ROOT, timeout=300)


def _grades(tmp_path, mapping):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"tag": "blocked",
                             "scale": "likelihood_same_percent",
                             "labels": {str(k): v for k, v in mapping.items()}}),
                 encoding="utf-8")
    return str(p)


def _rows():
    return json.load(open(SAMPLE, encoding="utf-8"))["pairs"]


def test_all_zero_grades_estimate_zero_hidden_merges():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        g = _grades(pathlib.Path(td), {i: 0 for i in range(len(_rows()))})
        r = _run(g)
        assert r.returncode == 0, r.stderr
        assert "point estimate            : 0" in r.stdout


def test_all_certain_grades_estimate_the_whole_population():
    """Weights sum to the population, so grading everything 100 must return it."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        g = _grades(pathlib.Path(td), {i: 100 for i in range(len(_rows()))})
        r = _run(g)
        assert r.returncode == 0, r.stderr
        assert "3,013,215" in r.stdout


def test_a_moved_sample_is_refused(tmp_path):
    """The position-keying guard. Must fail loudly, not estimate."""
    import shutil
    backup = str(tmp_path / "bak")
    shutil.copy2(SAMPLE, backup)
    try:
        with open(SAMPLE, "ab") as fh:
            fh.write(b" ")
        g = _grades(tmp_path, {0: 50})
        r = _run(g)
        assert r.returncode != 0
        assert "SAMPLE HAS CHANGED" in (r.stdout + r.stderr)
    finally:
        shutil.copy2(backup, SAMPLE)
    assert _run(_grades(tmp_path, {0: 50})).returncode == 0


def test_grade_index_outside_the_sample_is_refused(tmp_path):
    g = _grades(tmp_path, {99999: 100})
    r = _run(g)
    assert r.returncode != 0
    assert "outside the sample" in (r.stdout + r.stderr)


def test_ungraded_strata_are_reported_not_counted_as_zero(tmp_path):
    """Grading one row must not imply the rest of the pool is empty."""
    g = _grades(tmp_path, {0: 100})
    r = _run(g)
    assert r.returncode == 0, r.stderr
    assert "UNREPRESENTED" in r.stdout
    assert "not counted as zero" in r.stdout


def test_partial_grading_does_not_scale_to_the_whole_pool(tmp_path):
    """One row graded 100 must not report the full 3,013,215."""
    g = _grades(tmp_path, {0: 100})
    r = _run(g)
    line = [l for l in r.stdout.splitlines() if "point estimate" in l][0]
    est = int(line.split(":")[1].split()[0].replace(",", ""))
    assert est < 3013215
