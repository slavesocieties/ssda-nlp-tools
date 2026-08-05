"""Regressions from the 2026-08-05 review pass over that day's new tools.

The review found a class of defect, not a list of instances: several tools
accepted an EMPTY corpus, produced empty output, reported success, and
OVERWROTE existing artifacts with it. Running the check destroyed e1 and
v9tradeoff -- real results, gone, with a zero exit code.
"""
import json
import os

import pytest


@pytest.mark.parametrize("mod", ["analyze_surname_tradeoff", "run_evidence_merge",
                                 "build_targeted_labels", "calibrate_evidence"])
def test_empty_corpus_is_refused_not_written(mod, tmp_path):
    """An empty input is always a mistake -- a wrong --assembled path, an
    unfinished rebuild -- and never a result worth writing."""
    m = __import__(mod)
    with pytest.raises(SystemExit) as e:
        m.require_corpus([], str(tmp_path))
    assert "Refusing" in str(e.value)


@pytest.mark.parametrize("mod", ["analyze_surname_tradeoff", "run_evidence_merge",
                                 "build_targeted_labels", "calibrate_evidence"])
def test_a_populated_corpus_passes_through(mod, tmp_path):
    m = __import__(mod)
    assert m.require_corpus([{"id": "x"}], str(tmp_path)) == [{"id": "x"}]


def test_label_file_of_the_wrong_shape_says_so(tmp_path):
    """synthetic_labels.json stores PEOPLE; labels.json stores POINTERS. Feeding
    the first to verify_label_scores raised KeyError from inside the loop."""
    from verify_label_scores import score
    with pytest.raises(SystemExit) as e:
        score({}, {"a": {"name": "Ana"}, "b": {"name": "Ana"}})
    assert "PEOPLE, not pointers" in str(e.value)


def test_pointer_shaped_labels_still_resolve():
    from verify_label_scores import score
    s, why = score({}, {"a": {"entry": "E1", "id": "P01"},
                        "b": {"entry": "E2", "id": "P02"}})
    assert s is None and why == ["unresolved"]
