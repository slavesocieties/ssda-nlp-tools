"""Offline tests for ssda_nlp_tools — no API keys, no network.

Run:  python -m pytest tests -q     (from the repo root)
"""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import textmatch as tm
from ssda_nlp_tools import evaluate as ev
from ssda_nlp_tools import disambiguate as dis
from ssda_nlp_tools import fixes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(ROOT, "training_data.json")


# --------------------------------------------------------------------------- #
# textmatch
# --------------------------------------------------------------------------- #

def test_normalize_name_strips_accents_and_titles():
    assert tm.normalize_name("Don José Fernández") == "jose fernandez"
    assert tm.normalize_name("Doña María") == "maria"


def test_name_similarity_bounds():
    assert tm.name_similarity("Juan Vives", "Juan Vives") == 1.0
    assert tm.name_similarity("Matanzas", "Matansas") > 0.7   # spelling drift
    assert tm.name_similarity("Juan", "Pedro") == 0.0
    assert tm.name_similarity("", "x") == 0.0


def test_greedy_align_one_to_one():
    gold = [{"name": "Juan Vives"}, {"name": "Maria Ortega"}]
    pred = [{"name": "Maria Ortega"}, {"name": "Juan Vivez"}, {"name": "Extra Person"}]
    matches, ug, up = tm.greedy_align(gold, pred)
    assert len(matches) == 2 and ug == [] and len(up) == 1


def test_prf_edges():
    assert tm.prf(0, 0, 0)["f1"] == 1.0     # nothing expected, nothing predicted
    assert tm.prf(1, 1, 1)["precision"] == 0.5


# --------------------------------------------------------------------------- #
# evaluate  (uses the real gold set)
# --------------------------------------------------------------------------- #

def test_gold_vs_gold_is_perfect():
    gold = json.load(open(GOLD, encoding="utf-8"))
    r = ev.evaluate(gold, gold)
    assert r["people"]["f1"] == 1.0
    assert r["events"]["f1"] == 1.0
    assert r["relationships"]["f1"] == 1.0


def test_dropping_people_lowers_recall():
    gold = json.load(open(GOLD, encoding="utf-8"))
    pert = copy.deepcopy(gold)
    for ex in pert["examples"]:
        ppl = ex.get("data", {}).get("people", [])
        if len(ppl) >= 2:
            ppl.pop()
    r = ev.evaluate(gold, pert)
    assert r["people"]["recall"] < 1.0
    assert r["people"]["precision"] == 1.0   # remaining preds still correct


def test_attribute_flip_is_detected():
    gold = json.load(open(GOLD, encoding="utf-8"))
    pert = copy.deepcopy(gold)
    flipped = 0
    for ex in pert["examples"]:
        for p in ex.get("data", {}).get("people", []):
            if p.get("phenotype"):
                p["phenotype"] = "ZZZ"
                flipped += 1
    r = ev.evaluate(gold, pert)
    assert flipped > 0
    assert r["attributes"]["phenotype"]["accuracy"] < 1.0


def test_empty_predictions_zero_recall():
    gold = json.load(open(GOLD, encoding="utf-8"))
    empty = {"examples": [{"entry": e["entry"], "data": {"people": [], "events": []}}
                          for e in gold["examples"]]}
    r = ev.evaluate(gold, empty)
    assert r["people"]["recall"] == 0.0
    assert r["people"]["tp"] == 0


def test_norm_value_bool_and_accent():
    assert ev.norm_value(True) == "true"
    assert ev.norm_value("Libre") == "libre"
    assert ev.norm_value("negrá") == "negra"
    assert ev.norm_value(None) is None


# --------------------------------------------------------------------------- #
# disambiguate
# --------------------------------------------------------------------------- #

def _volume_with_repeated_priest():
    priest = {"id": "P01", "name": "Miguel O'Reilly", "occupation": "cleric", "titles": ["Don"]}
    kids = ["Juana Ramirez", "Pedro Alvarez", "Ana Gutierrez", "Tomas Delgado"]
    entries = []
    for i, kid in enumerate(kids):
        entries.append({"id": f"0013-0{i}", "data": {"people": [
            dict(priest),
            {"id": "P02", "name": kid, "age": "infant"},
        ], "events": []}})
    return {"id": 239746, "type": "baptism", "entries": entries}


def test_disambiguation_merges_recurring_person():
    vol = _volume_with_repeated_priest()
    res = dis.disambiguate_volume(vol)
    priests = [i for i in res["identities"] if "reilly" in tm.normalize_name(i["canonical_name"])]
    assert len(priests) == 1
    assert priests[0]["n_mentions"] == 4          # all four merged
    # the four distinct children must NOT be merged
    assert res["stats"]["identities"] == 5        # 1 priest + 4 children


def test_disambiguation_never_merges_within_entry():
    vol = {"id": 1, "entries": [{"id": "0001-01", "data": {"people": [
        {"id": "P01", "name": "Juan Perez"},
        {"id": "P02", "name": "Juan Perez"},   # two same-named people in ONE entry
    ], "events": []}}]}
    res = dis.disambiguate_volume(vol)
    assert res["stats"]["identities"] == 2        # stay separate


def test_sacrament_principal_guard_blocks_two_baptizees():
    # Two entries each baptizing an infant with the SAME name: these are two
    # different children (you are baptized once) -> must NOT auto-merge.
    vol = {"id": 1, "entries": [
        {"id": "0001-01", "data": {
            "people": [{"id": "P01", "name": "Maria Dolores", "age": "infant"}],
            "events": [{"type": "baptism", "principals": ["P01"]}]}},
        {"id": "0002-01", "data": {
            "people": [{"id": "P01", "name": "Maria Dolores", "age": "infant"}],
            "events": [{"type": "baptism", "principals": ["P01"]}]}},
    ]}
    res = dis.disambiguate_volume(vol)
    assert res["stats"]["identities"] == 2          # kept separate
    assert res["stats"]["merged_identities"] == 0
    blocked = [r for r in res["review_queue"]
               if any("sacrament" in x for x in r["reasons"])]
    assert len(blocked) == 1                         # but visible for review


def test_sacrament_guard_does_not_block_recurring_godparent():
    # A godparent (never a baptism principal) recurring across entries must
    # still merge even though each entry has its own baptizee.
    vol = {"id": 1, "entries": [
        {"id": f"000{i}-01", "data": {
            "people": [
                {"id": "P01", "name": f"Nino Distinto{i}", "age": "infant"},
                {"id": "P02", "name": "Isabel de los Rios",
                 "relationships": [{"related_person": "P01",
                                    "relationship_type": "godparent"}]},
            ],
            "events": [{"type": "baptism", "principals": ["P01"]}]}}
        for i in range(3)
    ]}
    res = dis.disambiguate_volume(vol)
    god = [x for x in res["identities"] if "isabel" in tm.normalize_name(x["canonical_name"])]
    assert len(god) == 1 and god[0]["n_mentions"] == 3


def test_estate_surname_does_not_link_different_spouses():
    # Everyone on an estate shares its surname; the GIVEN name must decide.
    assert dis._third_party_same("hanna macqueen", "rachael macqueen") is False
    assert dis._third_party_same("rachael", "rachael macqueen") is True     # short form
    assert dis._third_party_same("dafney macqueen", "dafiny macqueen") is True  # drift


def test_bare_name_with_conflicting_enslaver_context_does_not_merge():
    # "Juan, slave of Sanchez" and "Juan, slave of McQueen" are different people.
    def entry(eid, enslaver):
        return {"id": eid, "data": {"people": [
            {"id": "P01", "name": "Juan", "phenotype": "negro", "free": False,
             "relationships": [{"related_person": "P02", "relationship_type": "enslaver"}]},
            {"id": "P02", "name": enslaver},
        ], "events": []}}
    vol = {"id": 1, "entries": [entry("0001-01", "Francisco Sanchez"),
                                entry("0002-01", "Juan Macqueen")]}
    res = dis.disambiguate_volume(vol)
    juans = [i for i in res["identities"]
             if tm.normalize_name(i["canonical_name"]) == "juan"]
    assert len(juans) == 2 and all(j["n_mentions"] == 1 for j in juans)


def test_bare_name_with_shared_spouse_context_does_merge():
    def entry(eid, child):
        return {"id": eid, "data": {"people": [
            {"id": "P01", "name": "Smart", "phenotype": "negro", "free": False,
             "relationships": [{"related_person": "P02", "relationship_type": "spouse"}]},
            {"id": "P02", "name": "Rachael Macqueen"},
            {"id": "P03", "name": child, "age": "infant"},
        ], "events": [{"type": "baptism", "principals": ["P03"]}]}}
    vol = {"id": 1, "entries": [entry("0001-01", "Sara"), entry("0002-01", "Andres")]}
    res = dis.disambiguate_volume(vol)
    smarts = [i for i in res["identities"]
              if tm.normalize_name(i["canonical_name"]) == "smart"]
    assert len(smarts) == 1 and smarts[0]["n_mentions"] == 2


def test_hard_conflict_routes_to_review_not_merge():
    vol = {"id": 1, "entries": [
        {"id": "0001-01", "data": {"people": [{"id": "P01", "name": "Ana", "free": True}],
                                   "events": []}},
        {"id": "0002-01", "data": {"people": [{"id": "P01", "name": "Ana", "free": False}],
                                   "events": []}},
    ]}
    res = dis.disambiguate_volume(vol)
    assert res["stats"]["merged_identities"] == 0        # conflict blocks auto-merge
    assert res["stats"]["review_pairs"] == 1             # but shows up for a human


# --------------------------------------------------------------------------- #
# fixes  (regression guards vs the original bugs)
# --------------------------------------------------------------------------- #

def test_parse_date_returns_ints():
    assert fixes.parse_date("1873-01-17") == [1873, 1, 17]
    assert all(isinstance(x, int) for x in fixes.parse_date("1873-01-17"))


def test_parse_date_handles_range():
    assert fixes.parse_date("1873-01/1873-02") == [1873, 1, 1873, 2]


def test_complete_date_no_longer_crashes_on_string_month():
    assert fixes.complete_date("1873-01", "m") == (1873, 1, 1, 1873, 1, 31)
    assert fixes.complete_date("1873-01", "e") == (1873, 1, 31)


def test_is_principal_checks_all_events():
    events = [{"type": "baptism", "principals": ["P08"]},
              {"type": "birth", "principals": ["P08"]}]
    assert fixes.is_principal("P08", events) is True
    assert fixes.is_principal("P02", events) is False


def test_fix_relationships_adds_missing_reciprocal():
    data = {
        "people": [
            {"id": "P01", "name": "Parent", "relationships": [
                {"related_person": "P02", "relationship_type": "parent"}]},
            {"id": "P02", "name": "Child", "relationships": []},
        ],
        "events": [{"type": "baptism", "principals": ["P02"]}],
    }
    fixed, changes = fixes.fix_relationships(data)
    child = next(p for p in fixed["people"] if p["id"] == "P02")
    assert any(r["relationship_type"] == "child" and r["related_person"] == "P01"
               for r in child["relationships"])
    assert changes


def test_fix_relationships_multi_event_principal():
    # P02 is a principal only via the SECOND event; the buggy original (events[0]
    # only) would treat it as non-principal and mis-handle the fix.
    data = {
        "people": [
            {"id": "P01", "name": "Enslaver", "relationships": [
                {"related_person": "P02", "relationship_type": "enslaver"}]},
            {"id": "P02", "name": "Enslaved", "relationships": [
                {"related_person": "P01", "relationship_type": "godparent"}]},
        ],
        "events": [{"type": "baptism", "principals": ["P03"]},
                   {"type": "birth", "principals": ["P02"]}],
    }
    fixed, changes = fixes.fix_relationships(data)
    # conflicting types (enslaver vs godparent) with P02 a (second-event) principal:
    # resolved deterministically without raising
    assert isinstance(fixed, dict) and "people" in fixed
