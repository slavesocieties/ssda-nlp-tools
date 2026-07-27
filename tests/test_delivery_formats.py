"""Delivery-format gaps Daniel raised on 2026-07-24:

  * he found "indications that a network graph is being assembled" but no
    obvious nodes/edges files -> flat CSVs beside the GraphML
  * `witnesses` is specified in training_data_documentation.txt but was absent
    from our schema, so it was never extracted and never scored
"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import evaluate as E  # noqa: E402
from ssda_nlp_tools.network import to_csv  # noqa: E402


def _net():
    return {
        "nodes": [
            {"id": "n1", "label": "Miguel O'Reilly", "mentions": 21,
             "needs_review": False, "occupation": "Cleric", "free": True},
            {"id": "n2", "label": "María Dolores", "mentions": 3,
             "needs_review": True, "phenotype": "morena", "free": False},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "godparent", "weight": 2,
             "entries": ["e-01", "e-02"]},
        ],
        "stats": {},
    }


def test_csv_export_round_trips_nodes_and_edges():
    d = tempfile.mkdtemp()
    npath, epath = to_csv(_net(), os.path.join(d, "nodes.csv"),
                          os.path.join(d, "edges.csv"))

    with open(npath, encoding="utf-8") as f:
        nodes = list(csv.DictReader(f))
    assert [n["id"] for n in nodes] == ["n1", "n2"]
    assert nodes[0]["label"] == "Miguel O'Reilly"
    assert nodes[0]["mentions"] == "21"
    assert nodes[0]["occupation"] == "Cleric"
    assert nodes[1]["phenotype"] == "morena"

    with open(epath, encoding="utf-8") as f:
        edges = list(csv.DictReader(f))
    assert len(edges) == 1
    e = edges[0]
    assert (e["source"], e["target"], e["relationship_type"]) == ("n1", "n2", "godparent")
    assert e["weight"] == "2"
    # both endpoint labels are carried so the file is readable on its own
    assert e["source_label"] == "Miguel O'Reilly" and e["target_label"] == "María Dolores"
    assert e["entries"] == "e-01;e-02"


def test_csv_writes_booleans_as_true_false_not_python_repr():
    d = tempfile.mkdtemp()
    npath, _ = to_csv(_net(), os.path.join(d, "n.csv"), os.path.join(d, "e.csv"))
    text = open(npath, encoding="utf-8").read()
    assert "True" not in text and "False" not in text
    assert "true" in text and "false" in text


def test_missing_node_attributes_become_empty_cells_not_none():
    d = tempfile.mkdtemp()
    npath, _ = to_csv(_net(), os.path.join(d, "n.csv"), os.path.join(d, "e.csv"))
    with open(npath, encoding="utf-8") as f:
        nodes = list(csv.DictReader(f))
    assert nodes[0]["phenotype"] == ""
    assert "None" not in open(npath, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# witnesses
# --------------------------------------------------------------------------- #

def _people():
    return [{"id": "P01", "name": "Juan Perez"}, {"id": "P02", "name": "Ana Diaz"},
            {"id": "P03", "name": "Luis Mora"}, {"id": "P04", "name": "Rosa Gil"}]


def _rec(witnesses):
    ev = {"type": "marriage", "principals": ["P01", "P02"], "date": "1798-06-11"}
    if witnesses is not None:
        ev["witnesses"] = witnesses
    return [{"id": "e1", "data": {"people": _people(), "events": [ev]}}]


def test_missing_witness_is_caught_without_failing_the_event_match():
    """A marriage is the same marriage whether or not we caught its witnesses —
    so the event must still match, and the loss must still be visible."""
    rep = E.evaluate(_rec(["P03"]), _rec(None))
    assert rep["events"]["f1"] == 1.0
    w = rep["witnesses_on_matched_events"]
    assert w["fn"] == 1 and w["tp"] == 0


def test_correct_witnesses_score_clean():
    rep = E.evaluate(_rec(["P03", "P04"]), _rec(["P03", "P04"]))
    w = rep["witnesses_on_matched_events"]
    assert (w["tp"], w["fp"], w["fn"]) == (2, 0, 0)
    assert w["f1"] == 1.0


def test_invented_witness_is_a_false_positive():
    rep = E.evaluate(_rec(["P03"]), _rec(["P03", "P04"]))
    w = rep["witnesses_on_matched_events"]
    assert w["tp"] == 1 and w["fp"] == 1


def test_unexercised_witness_field_reports_none_not_zero():
    """Daniel's own gold has 24 events and 0 witnesses. Scoring that as 0.0 would
    read as a failure when the field simply never comes up."""
    rep = E.evaluate(_rec(None), _rec(None))
    assert rep["witnesses_on_matched_events"] is None
    assert "unexercised" in E.format_report(rep)
