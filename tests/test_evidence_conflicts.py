"""Conflicting relationships, graded by role. Daniel's rulings as tests.

2026-08-07
  "if two people with the same name have spouses with different names, that's
  essentially disqualifying - it's not just that they have no shared
  relationship, they have actively distinguishing individual relationships."

2026-08-10
  "Clearly different parents = automatically disqualifying; Clearly different
  spouses in an early modern Catholic society = essentially automatically
  disqualifying; different slave owners within a short span of time =
  essentially automatically disqualifying; different slave owner across a span
  of a decade+ = substantial penalty but not disqualifying."

  "children/godchildren can't really conflict because they're not a limited set,
  godparents/grandparents function similarly to parents, witnesses can only
  increase match probability through an overlapping association."
"""
import math

import pytest

from ssda_nlp_tools.evidence import (DISQUALIFY_MARGIN, ENSLAVER_SALE_YEARS,
                                     LOG_PRIOR_ODDS, MAX_HOLDERS,
                                     REVIEW_LOG_ODDS, W_CONFLICT_DISQUALIFYING,
                                     W_CONFLICT_SUBSTANTIAL, NameStats,
                                     _MAX_AGREEMENT, network_llr, score)


def _stats(names):
    return NameStats([{"name": n} for n in names])


def _m(name, year=None, **ctx):
    """ctx values may be one name or a tuple of names for the same role."""
    pairs = set()
    for role, val in ctx.items():
        for n in ((val,) if isinstance(val, str) else val):
            pairs.add((role, n))
    return {"name": name, "_year": year, "_ctx": pairs}


def _conflicts(a, b, s):
    return [w for w in network_llr(a, b, s)[1] if w.startswith("conflict")]


def _pen(a, b, s):
    return sum(float(w.split("(")[1].rstrip(")")) for w in _conflicts(a, b, s))


# --- spouse: indissoluble marriage ----------------------------------------

def test_a_different_spouse_is_disqualifying_not_merely_costly():
    s = _stats(["maria"] * 40 + ["juan perez", "pedro gomez"])
    assert _pen(_m("maria", 1850, spouse="juan perez"),
                _m("maria", 1852, spouse="pedro gomez"),
                s) == pytest.approx(W_CONFLICT_DISQUALIFYING, abs=5e-3)


def test_a_spouse_conflict_does_not_decay_with_time():
    """The first version divided the penalty by 3 past 15 years, on the theory
    that widowhood makes remarriage ordinary. Daniel graded cfl-009 -- a
    16-year gap that received exactly that discount -- as 0. The lawful second
    marriage is recorded separately as `former spouse`, so no discount is owed
    here."""
    s = _stats(["maria"] * 40 + ["juan perez", "pedro gomez"])
    near = _pen(_m("maria", 1850, spouse="juan perez"),
                _m("maria", 1852, spouse="pedro gomez"), s)
    far = _pen(_m("maria", 1800, spouse="juan perez"),
               _m("maria", 1860, spouse="pedro gomez"), s)
    assert near == far


def test_a_former_spouse_is_not_a_clash():
    """It is the register's own record of widowhood -- the one lawful way to
    have had two spouses. Penalising it would punish the very evidence that
    explains the second marriage."""
    s = _stats(["maria"] * 40 + ["juan perez", "pedro gomez"])
    a = _m("maria", 1850, **{"former spouse": "juan perez"})
    b = _m("maria", 1860, spouse="pedro gomez")
    assert "former spouse" not in MAX_HOLDERS
    assert _conflicts(a, b, s) == []


# --- enslaver: the one clash time can explain ------------------------------

def test_enslaver_conflict_is_graded_by_time_because_people_were_sold():
    s = _stats(["rita"] * 40 + ["gaston enriquez", "domingos lopes"])
    near = _pen(_m("rita", 1850, enslaver="gaston enriquez"),
                _m("rita", 1852, enslaver="domingos lopes"), s)
    far = _pen(_m("rita", 1800, enslaver="gaston enriquez"),
               _m("rita", 1800 + ENSLAVER_SALE_YEARS + 5,
                  enslaver="domingos lopes"), s)
    assert near == pytest.approx(W_CONFLICT_DISQUALIFYING, abs=5e-3)
    assert far == pytest.approx(W_CONFLICT_SUBSTANTIAL, abs=5e-3)
    assert far > near, "a decade of separation must soften it, not harden it"


def test_an_undated_enslaver_clash_gets_the_strict_reading():
    """With no dates there is no decade to appeal to, so it cannot claim the
    softer penalty."""
    s = _stats(["rita"] * 40 + ["gaston enriquez", "domingos lopes"])
    assert _pen(_m("rita", None, enslaver="gaston enriquez"),
                _m("rita", None, enslaver="domingos lopes"),
                s) == pytest.approx(W_CONFLICT_DISQUALIFYING, abs=5e-3)


# --- limited sets, and the capacity that defines them ----------------------

def test_one_parent_each_is_a_mother_and_a_father_not_a_conflict():
    """The false positive the A/B audit caught: two 1742 entries for one "Maria
    de Jesus" naming the SAME husband, split because one recorded the mother and
    the other the father."""
    s = _stats(["josefa"] * 30 + ["ana lopez", "ines vega"])
    assert _conflicts(_m("josefa", 1850, parent="ana lopez"),
                      _m("josefa", 1851, parent="ines vega"), s) == []


def test_three_distinct_parents_between_them_is_impossible():
    s = _stats(["josefa"] * 30 + ["ana lopez", "juan lopez", "ines vega"])
    a = _m("josefa", 1850, parent=("ana lopez", "juan lopez"))
    b = _m("josefa", 1851, parent="ines vega")
    assert _pen(a, b, s) == pytest.approx(W_CONFLICT_DISQUALIFYING, abs=5e-3)


def test_godparents_and_grandparents_are_limited_like_parents():
    assert MAX_HOLDERS["godparent"] == 2
    assert MAX_HOLDERS["grandparent"] == 4
    s = _stats(["luiza"] * 30 + ["ana lopez", "juan lopez", "ines vega"])
    a = _m("luiza", 1742, godparent=("ana lopez", "juan lopez"))
    b = _m("luiza", 1743, godparent="ines vega")
    assert _conflicts(a, b, s)


def test_children_godchildren_and_witnesses_can_never_conflict():
    """Not a limited set, so there is no limit to exceed."""
    s = _stats(["maria"] * 40 + ["ana lopez", "juan lopez", "ines vega",
                                 "rosa aulet", "pedro gomez", "clara roza"])
    for role in ("child", "godchild", "witness", "grandchild", "sibling"):
        assert role not in MAX_HOLDERS
        a = _m("maria", 1850, **{role: ("ana lopez", "juan lopez", "ines vega")})
        b = _m("maria", 1851, **{role: ("rosa aulet", "pedro gomez", "clara roza")})
        assert _conflicts(a, b, s) == [], f"{role} must never conflict"


def test_unlimited_roles_do_not_drive_the_disjointness_penalty_either():
    """Counting them punished a record for naming more people, which is
    backwards -- richer records were penalised for being richer."""
    s = _stats(["maria"] * 40 + ["ana lopez", "juan lopez", "ines vega",
                                 "rosa aulet"])
    a = _m("maria", 1850, child=("ana lopez", "juan lopez"))
    b = _m("maria", 1851, child=("ines vega", "rosa aulet"))
    llr, why = network_llr(a, b, s)
    assert not any("disjoint" in w for w in why)
    assert llr == 0.0


# --- the properties that make all of the above safe ------------------------

def test_a_spelling_variant_is_not_a_different_person():
    s = _stats(["maria"] * 40 + ["juan perez"])
    assert _conflicts(_m("maria", 1850, spouse="juan perez"),
                      _m("maria", 1852, spouse="juan peres"), s) == []


def test_a_shared_associate_is_still_positive_evidence():
    s = _stats(["maria"] * 40 + ["custodio vieira"])
    assert network_llr(_m("maria", 1850, spouse="custodio vieira"),
                       _m("maria", 1852, spouse="custodio vieira"), s)[0] > 0


def test_disqualifying_outweighs_the_strongest_possible_agreement():
    """What makes Daniel's "automatically" hold without a veto. The weight is
    DERIVED from this property, so this test is its definition -- if any other
    weight grows, this must still hold."""
    best = LOG_PRIOR_ODDS + _MAX_AGREEMENT + W_CONFLICT_DISQUALIFYING
    assert best <= REVIEW_LOG_ODDS - DISQUALIFY_MARGIN + 1e-9


def test_a_disqualifying_conflict_beats_a_perfect_name_and_a_shared_associate():
    """The concrete version of the above, through score() rather than arithmetic."""
    s = _stats(["maria"] * 3 + ["custodio jose vieira da silva",
                                "juan perez", "pedro gomez"])
    a = _m("maria", 1850, spouse="juan perez",
           godchild="custodio jose vieira da silva")
    b = _m("maria", 1851, spouse="pedro gomez",
           godchild="custodio jose vieira da silva")
    r = score(a, b, s)
    assert r["decision"] == "refuse"


def test_conflicts_are_penalties_never_vetoes():
    """The relation names come from an LLM and can be wrong, so even a
    disqualifying clash must stay finite and recoverable in principle."""
    s = _stats(["maria"] * 40 + ["juan perez", "pedro gomez"])
    r = score(_m("maria", 1850, spouse="juan perez"),
              _m("maria", 1852, spouse="pedro gomez"), s)
    assert r["vetoed"] is None
    assert math.isfinite(r["log_odds"])
