"""Offline tests for the review cycle: constraints, HTML generation, round-trip."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools.disambiguate import disambiguate_volume
from ssda_nlp_tools.review_html import decisions_to_constraints, render_review_html


def _vol_review_pair():
    """Two mentions that land in the review band (hard attr conflict)."""
    return {"id": 1, "entries": [
        {"id": "0001-01", "data": {"people": [{"id": "P01", "name": "Ana Fuentes",
                                               "free": True}], "events": []}},
        {"id": "0002-01", "data": {"people": [{"id": "P01", "name": "Ana Fuentes",
                                               "free": False}], "events": []}},
    ]}


def test_must_link_constraint_merges_review_pair():
    vol = _vol_review_pair()
    base = disambiguate_volume(vol)
    assert base["stats"]["identities"] == 2 and base["stats"]["review_pairs"] == 1
    cons = {"must": [[{"entry": "0001-01", "id": "P01"}, {"entry": "0002-01", "id": "P01"}]],
            "cannot": []}
    after = disambiguate_volume(vol, constraints=cons)
    assert after["stats"]["identities"] == 1          # human said: same person
    assert after["stats"]["review_pairs"] == 0        # decided -> out of the queue


def test_cannot_link_constraint_blocks_auto_merge():
    # identical names + agreeing attrs would auto-merge; a human says no
    vol = {"id": 1, "entries": [
        {"id": "0001-01", "data": {"people": [{"id": "P01", "name": "Pedro Gomez",
                                               "occupation": "soldier"}], "events": []}},
        {"id": "0002-01", "data": {"people": [{"id": "P01", "name": "Pedro Gomez",
                                               "occupation": "soldier"}], "events": []}},
    ]}
    assert disambiguate_volume(vol)["stats"]["identities"] == 1     # merges by default
    cons = {"must": [], "cannot": [[{"entry": "0001-01", "id": "P01"},
                                    {"entry": "0002-01", "id": "P01"}]]}
    after = disambiguate_volume(vol, constraints=cons)
    assert after["stats"]["identities"] == 2                        # kept apart


def test_decisions_to_constraints_mapping():
    decisions = {"decisions": [
        {"a": {"entry": "e1", "id": "P01"}, "b": {"entry": "e2", "id": "P01"},
         "decision": "same"},
        {"a": {"entry": "e3", "id": "P02"}, "b": {"entry": "e4", "id": "P02"},
         "decision": "different"},
        {"a": {"entry": "e5", "id": "P03"}, "b": {"entry": "e6", "id": "P03"},
         "decision": "unsure"},
    ]}
    cons = decisions_to_constraints(decisions)
    assert len(cons["must"]) == 1 and len(cons["cannot"]) == 1      # unsure ignored


def test_render_review_html_escapes_and_embeds(tmp_path):
    queue = [{"score": 0.8, "reasons": ["name~1.00"],
              "a": {"entry": "0001-01", "id": "P01", "name": "Eva </script> Fish",
                    "detail": {"phenotype": "negra"}},
              "b": {"entry": "0002-01", "id": "P01", "name": "Eva Fish", "detail": {}}}]
    path = str(tmp_path / "r.html")
    render_review_html(queue, path, tag="T")
    text = open(path, encoding="utf-8").read()
    assert "</script> Fish" not in text          # payload cannot break the script tag
    assert "\\u003c/script> Fish" in text        # it is escaped instead
    m = re.search(r"const PAIRS = (\[.*?\]);\nconst KEY", text, re.S)
    assert json.loads(m.group(1))[0]["a"]["name"] == "Eva </script> Fish"  # data intact
