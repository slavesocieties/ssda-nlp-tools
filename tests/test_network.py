"""Offline tests for phonetic matching, resolution, and the network build."""
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import textmatch as tm
from ssda_nlp_tools.resolve import resolve_volume
from ssda_nlp_tools.network import build_network, to_graphml


# ---- phonetic ---------------------------------------------------------------

def test_phonetic_collapses_scribal_variants():
    for a, b in [("Gonzalez", "Gonzales"), ("Vives", "Bibes"),
                 ("Matanzas", "Matansas"), ("Quintero", "Kintero")]:
        assert tm.phonetic_fold(a.lower()) == tm.phonetic_fold(b.lower()), (a, b)
        assert tm.name_similarity(a, b) >= 0.85


def test_phonetic_does_not_over_merge():
    assert tm.name_similarity("Juan", "Pedro") == 0.0
    assert tm.name_similarity("Ana Ruiz", "Luis Gomez") == 0.0


def test_phonetic_key_blocks_variants_together():
    assert tm.phonetic_key("Gonzalez Fernandez") == tm.phonetic_key("Gonzales Perez")


# ---- resolve ----------------------------------------------------------------

def _volume():
    priest = {"id": "P01", "name": "Miguel O'Reilly", "occupation": "cleric"}
    return {"id": 1, "type": "baptism", "entries": [
        {"id": "0001-01", "data": {"people": [
            dict(priest),
            {"id": "P02", "name": "Juana Ramirez",
             "relationships": [{"related_person": "P03", "relationship_type": "parent"}]},
            {"id": "P03", "name": "Pedro Ramirez",
             "relationships": [{"related_person": "P02", "relationship_type": "child"}]},
        ], "events": [{"type": "baptism", "principals": ["P03"]}]}},
        {"id": "0002-01", "data": {"people": [
            dict(priest),
            {"id": "P02", "name": "Ana Gutierrez"},
        ], "events": [{"type": "baptism", "principals": ["P02"]}]}},
    ]}


def test_resolve_annotates_global_ids_and_links_priest():
    res = resolve_volume(_volume())
    # every person mention got a global_id
    for e in res["volume"]["entries"]:
        for p in e["data"]["people"]:
            assert "global_id" in p
    # the priest in both entries resolves to the SAME global id
    g1 = res["volume"]["entries"][0]["data"]["people"][0]["global_id"]
    g2 = res["volume"]["entries"][1]["data"]["people"][0]["global_id"]
    assert g1 == g2


# ---- network ----------------------------------------------------------------

def test_network_builds_edges_and_is_graphml_valid(tmp_path):
    net = build_network(_volume())
    assert net["stats"]["nodes"] >= 4
    # parent/child edge exists between the two Ramirez
    types = {e["type"] for e in net["edges"]}
    assert "parent" in types and "child" in types
    path = str(tmp_path / "n.graphml")
    to_graphml(net, path)
    root = ET.parse(path).getroot()          # must be well-formed
    assert root.tag.endswith("graphml")


def test_network_reports_cross_entry_people():
    net = build_network(_volume())
    # the priest appears in 2 entries -> at least one cross-entry person
    assert net["stats"]["cross_entry_people"] >= 1


# ---- cross-chunk linking ------------------------------------------------------

def test_link_volumes_finds_cross_chunk_identity():
    from ssda_nlp_tools.link import link_volumes
    priest = {"id": "P01", "name": "Miguel O'Reilly", "occupation": "cleric"}
    chunk_a = {"id": 1, "entries": [
        {"id": "0001-01", "data": {"people": [dict(priest),
            {"id": "P02", "name": "Juana Ramirez", "age": "infant"}],
            "events": [{"type": "baptism", "principals": ["P02"]}]}}]}
    chunk_b = {"id": 1, "entries": [
        {"id": "0002-01", "data": {"people": [dict(priest),
            {"id": "P02", "name": "Pedro Alvarez", "age": "infant"}],
            "events": [{"type": "baptism", "principals": ["P02"]}]}}]}
    res = link_volumes([chunk_a, chunk_b], tags=["a", "b"], volume_tag="T")
    cross = [r for r in res["registry"] if r["cross_chunk"]]
    assert len(cross) == 1
    assert "reilly" in cross[0]["canonical_name"].lower()
    assert cross[0]["chunks"] == ["a", "b"]
    # the two baptizees must remain distinct, single-chunk identities
    assert res["stats"]["identities"] == 3


def test_combine_volumes_keeps_entry_ids_unique():
    from ssda_nlp_tools.link import combine_volumes
    v = {"id": 1, "entries": [{"id": "0001-01", "data": {"people": [], "events": []}}]}
    combined = combine_volumes([v, v], tags=["x", "y"])
    ids = [e["entry"] for e in combined["entries"]]
    assert len(ids) == len(set(ids)) == 2
    assert combined["entries"][0]["chunk"] == "x"
    assert combined["entries"][1]["chunk"] == "y"
