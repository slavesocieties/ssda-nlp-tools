"""Offline tests for the QA report."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools.qa import qa_volume

RAW = ("En diez y seis de Enero de mil setecientos ochenta y cuatro yo Don Thomas "
       "Hassett cura parroco bautice solemnemente a {kid} hijo legitimo de Smart y "
       "de Rachael esclavos de Don Juan Macqueen fueron padrinos Thomas Sterling")


def _entry(eid, kid, date="1784-01-16", raw=None):
    return {"id": eid, "raw": raw or RAW.format(kid=kid),
            "data": {"people": [{"id": "P01", "name": kid, "age": "infant"}],
                     "events": [{"type": "baptism", "principals": ["P01"], "date": date}]}}


def test_duplicate_requires_same_principal():
    # same-family formulaic text, DIFFERENT children -> not a duplicate
    vol = {"id": 1, "entries": [_entry("0001-01", "Sara"), _entry("0001-02", "Andres")]}
    rep = qa_volume(vol)
    assert not rep["duplicates"]

    # near-identical text, SAME child -> confirmed duplicate
    vol2 = {"id": 1, "entries": [_entry("0001-01", "Sara"), _entry("0001-02", "Sara")]}
    rep2 = qa_volume(vol2)
    assert len(rep2["duplicates"]) == 1
    assert rep2["duplicates"][0]["confidence"] == "confirmed"


def test_chronology_break_detected():
    vol = {"id": 1, "entries": [
        _entry("0001-01", "Sara", date="1784-05-01"),
        _entry("0002-01", "Andres", date="1783-01-01"),   # jumps back >30 days
    ]}
    rep = qa_volume(vol)
    assert len(rep["chronology_breaks"]) == 1


def test_impossible_date_detected():
    vol = {"id": 1, "entries": [_entry("0001-01", "Sara", date="1784-13-41")]}
    rep = qa_volume(vol)
    assert rep["issues_by_type"].get("impossible_date") == 1


def test_dangling_refs_detected():
    vol = {"id": 1, "entries": [{
        "id": "0001-01", "raw": "x",
        "data": {"people": [{"id": "P01", "name": "Sara",
                             "relationships": [{"related_person": "P99",
                                                "relationship_type": "parent"}]}],
                 "events": [{"type": "baptism", "principals": ["P77"]}]}}]}
    rep = qa_volume(vol)
    assert rep["issues_by_type"].get("dangling_relationship") == 1
    assert rep["issues_by_type"].get("dangling_principal") == 1


def test_event_shape_rules():
    vol = {"id": 1, "entries": [{
        "id": "0001-01", "raw": "x",
        "data": {"people": [{"id": "P01", "name": "A"}, {"id": "P02", "name": "B"}],
                 "events": [{"type": "marriage", "principals": ["P01"]}]}}]}
    rep = qa_volume(vol)
    assert rep["issues_by_type"].get("event_shape") == 1


def test_vocabulary_distributions_present():
    vol = {"id": 1, "entries": [_entry("0001-01", "Sara")]}
    rep = qa_volume(vol)
    assert "phenotype" in rep["vocabulary"]


def test_544367_segment_to_qa_keeps_distinct_trailing_record():
    """Regression over the real two-page fixture: No. 546 is a distinct trailing
    partial, not a page-boundary duplicate of No. 545. Representative extraction
    principals exercise the same principal-aware QA path used in production."""
    from ssda_nlp_tools.segment import load_pages, segment_volume

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "544367_sample.json")
    entries = segment_volume(load_pages(fixture))["entries"]
    principals = ["Manuel Salvador", "Francisco Miguel", "Victor Clemente",
                  "Segundo Guillermo", "Juan Francisco", "Juan Alberto"]
    assert len(entries) == len(principals) == 6
    for entry, principal in zip(entries, principals):
        entry["data"] = {
            "people": [{"id": "P01", "name": principal}],
            "events": [{"type": "baptism", "principals": ["P01"]}],
        }

    rep = qa_volume({"id": "544367", "entries": entries})
    assert rep["entries"] == 6
    assert rep["duplicates"] == []
