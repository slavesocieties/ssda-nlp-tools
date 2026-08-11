"""Reading the family side off the register's own words.

Daniel, 2026-08-10: "This is a question that can/should be resolved upstream.
Maternal/paternal grandparents should be labeled differently when extracted."

The side is already in the transcription in 97.8% of the entries that carry a
grandparent edge, so these tests are written against the actual formulae found
in the delivered volumes rather than invented sentences.
"""
import json
import os

import pytest

from ssda_nlp_tools.fixes import RECIPROCAL_RELS, fix_relationships, reciprocates
from ssda_nlp_tools.grandparent_side import (AMBIGUOUS, MATERNAL, NO_NAME,
                                             NO_SIDE_CLAUSE, NOT_NAMED,
                                             PATERNAL, classify, label_data,
                                             label_examples, side_clauses,
                                             side_of)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _record(text, grandparents):
    """A minimal baptism record: P01 the child, then one person per grandparent."""
    people = [{"id": "P01", "name": "Nino", "relationships": []}]
    for i, name in enumerate(grandparents, start=2):
        pid = f"P{i:02d}"
        people[0]["relationships"].append({"related_person": pid,
                                           "relationship_type": "grandparent"})
        people.append({"id": pid, "name": name,
                       "relationships": [{"related_person": "P01",
                                          "relationship_type": "grandchild"}]})
    return {"people": people,
            "events": [{"type": "baptism", "principals": ["P01"]}]}, text


def _sides(data):
    return [r["relationship_type"] for r in data["people"][0]["relationships"]]


# --- the formulae, as they actually appear ---------------------------------

def test_the_colon_list_form():
    """176899's commonest form. A colon is NOT a clause boundary -- treating it
    as one ended every clause of this shape one character after the cue."""
    data, text = _record(
        "Abuelos paternos: Cecilio Correa e Isabel Serrano, naturales de "
        "Trinidad, ambos difuntos; maternos: Buenviage Olivera, natural de "
        "Santa Clara.",
        ["Cecilio Correa", "Isabel Serrano", "Buenviage Olivera"])
    out, stats = label_data(data, text)
    assert _sides(out) == [PATERNAL, PATERNAL, MATERNAL]
    assert {k: stats[k] for k in ("seen", "maternal", "paternal", "unresolved")} \
        == {"seen": 3, "maternal": 1, "paternal": 2, "unresolved": 0}


def test_the_comma_and_semicolon_form():
    data, text = _record(
        "Abuelos paternos, los pardos Miguel Bazan y Belen de la Rosa, "
        "naturales de Camarones; maternos, los pardos Juan Lorenzo Morejon y "
        "Caridad Hernandez, vecinos de esta feligresia.",
        ["Miguel Bazan", "Belen de la Rosa", "Juan Lorenzo Morejon",
         "Caridad Hernandez"])
    assert _sides(label_data(data, text)[0]) == [PATERNAL, PATERNAL,
                                                 MATERNAL, MATERNAL]


def test_the_nieto_form_states_the_side_from_the_grandchild_s_view():
    """"nieto paterno de Jose y de Belen Sequeira y materno de Pio y de Rosario
    Marias" -- the noun is the grandchild's, the label is the grandparent's."""
    data, text = _record(
        "hijo legitimo de Rufino Sequeira y de Matilde Silva, nieto paterno de "
        "Jose y de Belen Sequeira y materno de Pio y de Rosario Marias.",
        ["Jose Sequeira", "Belen Sequeira", "Pio Marias", "Rosario Marias"])
    assert _sides(label_data(data, text)[0]) == [PATERNAL, PATERNAL,
                                                 MATERNAL, MATERNAL]


def test_a_single_grandmother_with_a_surname_the_clause_omits():
    """"Abuela materna Manuela criolla" -- extraction supplies a surname the
    clause does not, so the all-tokens pass fails and the given-name pass
    carries it."""
    data, text = _record("Abuela materna Manuela criolla.", ["Manuela Ferrer"])
    assert _sides(label_data(data, text)[0]) == [MATERNAL]


def test_portuguese_uses_the_same_adjectives():
    data, text = _record("Avos paternos Joao da Silva e Maria Antonia.",
                         ["Joao da Silva", "Maria Antonia"])
    assert _sides(label_data(data, text)[0]) == [PATERNAL, PATERNAL]


def test_accents_do_not_hide_the_cue():
    data, text = _record("Avós maternos: João Calvo e Serafina Pérez.",
                         ["João Calvo", "Serafina Pérez"])
    assert _sides(label_data(data, text)[0]) == [MATERNAL, MATERNAL]


# --- refusing to guess ------------------------------------------------------

def test_a_record_that_never_says_which_side_stays_unsided():
    """Guessing would fabricate a lineage. The bare term stays in the vocabulary
    precisely so there is somewhere honest to put these."""
    data, text = _record("Abuelos: Miguel Bazan y Belen de la Rosa.",
                         ["Miguel Bazan", "Belen de la Rosa"])
    out, stats = label_data(data, text)
    assert _sides(out) == ["grandparent", "grandparent"]
    assert stats["unresolved"] == 2


def test_a_name_falling_in_both_clauses_is_refused():
    """A surname shared across the two sides must not decide it."""
    data, text = _record(
        "Abuelos paternos Juan Correa y Ana Correa; maternos Juan Correa.",
        ["Juan Correa"])
    assert _sides(label_data(data, text)[0]) == ["grandparent"]


def test_a_grandparent_the_side_clauses_do_not_name_stays_unsided():
    """The real residue in the corpus: an extraction defect pointing a
    grandparent edge at the godparent. It must not inherit a side from the
    clause that happens to be nearest."""
    data, text = _record(
        "Abuelos maternos: Buenviage Olivera. Fueron sus padrinos Leonardo Rosa.",
        ["Leonardo Rosa"])
    assert _sides(label_data(data, text)[0]) == ["grandparent"]


def test_the_four_ways_it_can_fail_are_told_apart():
    """Daniel needs the register's silence separated from our own defects: the
    first two are nobody's fault, the last is almost always an extraction error
    (a grandparent edge pointing at the godparent), so lumping them into one
    "unresolved" count would hide a quality signal inside a coverage number.
    Verified on the corpus: every sampled `name_in_no_clause` case is a person
    the same entry also records as a godparent.
    """
    none_ = side_clauses("Abuelos: Miguel Bazan.")
    both = side_clauses("Abuelos paternos Juan Correa; maternos Juan Correa.")
    one = side_clauses("Abuela materna: Buenviage Olivera.")
    assert classify("Miguel Bazan", none_) == (None, NO_SIDE_CLAUSE)
    assert classify("", one) == (None, NO_NAME)
    assert classify("Juan Correa", both) == (None, AMBIGUOUS)
    assert classify("Leonardo Rosa", one) == (None, NOT_NAMED)
    assert classify("Buenviage Olivera", one) == (MATERNAL, None)


def test_the_reasons_reach_the_stats():
    data, text = _record(
        "Abuelos maternos: Buenviage Olivera. Fueron sus padrinos Leonardo Rosa.",
        ["Buenviage Olivera", "Leonardo Rosa"])
    _, stats = label_data(data, text)
    assert stats["maternal"] == 1 and stats["unresolved"] == 1
    assert stats[NOT_NAMED] == 1
    assert stats["unresolved_detail"] == [("Leonardo Rosa", NOT_NAMED)]


def test_the_godparents_are_outside_every_side_clause():
    clauses = side_clauses("Abuela materna: Desideria Espinosa, natural de San "
                           "Antonio. Fueron sus padrinos Joaquin Garcia y "
                           "Belen Acebal.")
    assert len(clauses) == 1
    assert "joaquin" not in clauses[0][1]
    assert side_of("Joaquin Garcia", clauses) is None


def test_no_text_means_no_side():
    data, _ = _record("", ["Miguel Bazan"])
    assert _sides(label_data(data, "")[0]) == ["grandparent"]


# --- reciprocity ------------------------------------------------------------

def test_all_three_terms_reciprocate_to_a_plain_grandchild():
    for term in ("grandparent", MATERNAL, PATERNAL):
        assert RECIPROCAL_RELS[term] == "grandchild"
        assert reciprocates(term, "grandchild")
    # ...and a grandchild edge is answered by any of them
    for term in ("grandparent", MATERNAL, PATERNAL):
        assert reciprocates("grandchild", term)
    assert not reciprocates("grandchild", "godparent")


def test_a_sided_pair_survives_the_reciprocity_fixer():
    """Before `reciprocates`, "P01 maternal grandparent P02" plus "P02
    grandchild P01" -- the correct pair -- read as a type mismatch and BOTH
    edges were dropped."""
    data, _ = _record("x", ["Manuela Ferrer"])
    data["people"][0]["relationships"][0]["relationship_type"] = MATERNAL
    fixed, changes = fix_relationships(data)
    assert _sides(fixed) == [MATERNAL]
    assert fixed["people"][1]["relationships"] == [
        {"related_person": "P01", "relationship_type": "grandchild"}]
    assert changes == []


def test_a_missing_reciprocal_is_added_unsided():
    """Repairing a one-sided "grandchild" edge cannot invent the side, so it
    adds the honest bare term."""
    data = {"people": [{"id": "P01", "name": "Abuela", "relationships": [
                            {"related_person": "P02",
                             "relationship_type": "grandchild"}]},
                       {"id": "P02", "name": "Nino", "relationships": []}],
            "events": [{"type": "baptism", "principals": ["P02"]}]}
    fixed, changes = fix_relationships(data)
    assert fixed["people"][1]["relationships"] == [
        {"related_person": "P01", "relationship_type": "grandparent"}]
    assert changes


# --- the few-shot pool ------------------------------------------------------

def test_daniels_gold_examples_are_relabelled_for_the_prompt():
    """Six of the fifteen carry grandparent edges and every source text states
    the side. Shown as-is they would demonstrate the opposite of what the prompt
    asks for -- and training_data.json is vendored verbatim from
    slavesocieties/openai, so this happens in memory, not in the file."""
    with open(os.path.join(_ROOT, "training_data.json"), encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    def terms(pool):
        return [r["relationship_type"] for ex in pool
                for p in ex["data"]["people"]
                for r in (p.get("relationships") or [])
                if "grandparent" in r["relationship_type"]]

    before, after = terms(examples), terms(label_examples(examples))
    assert before and set(before) == {"grandparent"}
    assert set(after) == {MATERNAL, PATERNAL}, "every gold example resolves"
    assert len(before) == len(after), "relabelled, never added or dropped"
    # the vendored file itself is untouched
    assert terms(examples) == before


def test_relabelling_is_deterministic_so_the_cache_prefix_holds():
    """build_messages relabels on every call; a non-deterministic result would
    break prompt caching, which is the whole cost model."""
    with open(os.path.join(_ROOT, "training_data.json"), encoding="utf-8") as f:
        examples = json.load(f)["examples"]
    a = json.dumps(label_examples(examples), sort_keys=True, ensure_ascii=False)
    b = json.dumps(label_examples(examples), sort_keys=True, ensure_ascii=False)
    assert a == b
