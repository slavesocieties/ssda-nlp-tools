"""The label re-scoring harness must fail loudly when it is degenerate.

The first version of this check reported that all ten of Daniel's negatives were
now correctly refused. That was false and flattering: labels.json stores pointers
with no "name" field, so pair_score short-circuited to 0.00 for every pair, and
"0.00 < 0.86" reads as "refused". These tests pin the control that caught it.
"""
import json
import os

import pytest

from verify_label_scores import main, merged_lookup, score


def _labels(tmp, rows):
    p = os.path.join(tmp, "labels.json")
    json.dump({"labels": rows}, open(p, "w", encoding="utf-8"))
    return p


def _pair(a_entry, a_id, b_entry, b_id, **kw):
    return {"a": {"entry": a_entry, "id": a_id},
            "b": {"entry": b_entry, "id": b_id},
            "names": [a_id, b_id], "score": kw.get("stored", 1.0),
            "likelihood": kw.get("likelihood")}


def test_unresolvable_pointers_score_none_not_zero():
    """The bug: an unresolved pointer must not look like a confident refusal."""
    s, reasons = score({}, _pair("e1", "P01", "e2", "P02"))
    assert s is None
    assert reasons == ["unresolved"]


def test_merged_lookup_groups_mentions_by_identity(tmp_path):
    p = os.path.join(str(tmp_path), "ids.json")
    json.dump([{"mentions": [{"entry": "e1", "id": "P01"},
                             {"entry": "e2", "id": "P02"}]},
               {"mentions": [{"entry": "e3", "id": "P01"}]}],
              open(p, "w", encoding="utf-8"))
    look = merged_lookup(p)
    assert look[("e1", "P01")] == look[("e2", "P02")]      # same identity
    assert look[("e1", "P01")] != look[("e3", "P01")]      # different identity


def test_missing_identities_file_returns_none(tmp_path):
    assert merged_lookup(os.path.join(str(tmp_path), "nope.json")) is None


def test_degenerate_harness_is_refused(tmp_path, monkeypatch, capsys):
    """Every pair scoring the same value means the harness is broken. It must
    refuse before printing a single verdict about Daniel's labels."""
    import verify_label_scores as V
    monkeypatch.setattr(V, "load_mentions", lambda a: ({}, 7))
    monkeypatch.setattr(V, "score", lambda idx, p: (0.0, ["different names"]))
    lab = _labels(str(tmp_path), [_pair("e1", "P01", "e2", "P02",
                                        stored=1.0, likelihood=100)] * 3)
    rc = main(["--labels", lab, "--identities", "nope.json"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "DEGENERATE" in out
    assert "Refusing to report verdicts" in out


def test_a_healthy_harness_passes_the_control(tmp_path, monkeypatch, capsys):
    import verify_label_scores as V
    monkeypatch.setattr(V, "load_mentions", lambda a: ({}, 7))
    scores = iter([0.95, 0.88, 1.00] * 20)
    monkeypatch.setattr(V, "score", lambda idx, p: (next(scores), ["name~1.00"]))
    lab = _labels(str(tmp_path), [_pair("e1", "P01", "e2", "P02",
                                        stored=0.95, likelihood=100),
                                  _pair("e3", "P01", "e4", "P02",
                                        stored=0.95, likelihood=0)])
    rc = main(["--labels", lab, "--identities", "nope.json"])
    assert rc == 0
    assert "harness OK" in capsys.readouterr().out
