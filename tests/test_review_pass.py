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


# --- arithmetic in the analysis scripts -------------------------------------

def test_transatlantic_uses_real_distance_not_volume_difference(tmp_path):
    """A Cuba+Brazil cluster must flag; two Havana parishes 6km apart must not,
    even though both are 'more than one volume'."""
    from compare_scorers import transatlantic
    from ssda_nlp_tools.volume_geo import load
    geo = load()
    if geo is None:
        pytest.skip("volumes.json unavailable")
    ids = [{"canonical_name": "Far", "n_mentions": 2,
            "mentions": [{"entry": "201991-0001-01", "id": "P01"},
                         {"entry": "701179-0002-01", "id": "P01"}]},
           {"canonical_name": "Near", "n_mentions": 2,
            "mentions": [{"entry": "201991-0004-01", "id": "P01"},
                         {"entry": "29597-0005-01", "id": "P01"}]},
           {"canonical_name": "Single", "n_mentions": 1,
            "mentions": [{"entry": "201991-0006-01", "id": "P01"}]}]
    got = transatlantic(ids, geo)
    assert [g[0] for g in got] == ["Far"]


def test_review_load_is_not_labelled_as_a_percentage():
    """It prints PAIRS PER MENTION. An earlier version put that under a heading
    quoting Daniel's '.1% acceptable, 10% is not' -- 16.0 pairs per mention is
    not 16%, it is 1600%, and the juxtaposition invites a wrong reading."""
    src = open("compare_scorers.py", encoding="utf-8").read()
    head = src[src.index("4. REVIEW"):src.index("return 0")]
    assert "REVIEW RATE" not in head, "'rate' implies a percentage"
    assert "NOT" in head and "comparable" in head


def test_self_check_detects_an_emptied_artifact(tmp_path):
    """It previously SKIPPED a corrupted run, because its A/B rule only compares
    runs of equal mention count and nothing else had zero."""
    import self_check
    d = tmp_path / "production" / "luna_v3" / "merge"
    d.mkdir(parents=True)
    json.dump({"mentions": 0, "identities": 0}, open(d / "dead.stats.json", "w"))
    ok, msg = self_check._artifacts_intact(str(tmp_path))
    assert ok is False and "dead.stats.json" in msg


def test_self_check_passes_on_healthy_artifacts(tmp_path):
    import self_check
    d = tmp_path / "production" / "luna_v3" / "merge"
    d.mkdir(parents=True)
    json.dump({"mentions": 39697, "identities": 33099}, open(d / "good.stats.json", "w"))
    ok, _ = self_check._artifacts_intact(str(tmp_path))
    assert ok is True
