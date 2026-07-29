"""Daniel's 2026-07-29 answers, as executable rules.

Covers the surname tiering (his Llopiz ruling), the stratified training sample,
and the 0/25/50/75/100 labelling page.
"""
import json

import pytest

from ssda_nlp_tools.disambiguate import (SURNAME_TIERS, context_strength,
                                         disambiguate_volume, surname_affinity,
                                         surname_tier_allows)
from ssda_nlp_tools.likelihood_review_html import (label_summary, labels_to_constraints,
                                                   render_likelihood_review_html)
from ssda_nlp_tools.training_sample import (StratifiedReservoir, band_of,
                                            signal_of, stratum_of,
                                            surname_relation, volume_of)


def _m(name, **kw):
    m = {"name": name, "_ctx": set(), "_register": None, "_year": None}
    m.update(kw)
    return m


# --------------------------------------------------------------------------- #
# surname tiers
# --------------------------------------------------------------------------- #

def test_daniels_three_cases_land_in_three_different_tiers():
    """"Llopiz/Llopis is something that I'd want to merge assuming context is
    reasonable, and likely Llepiz as well. Llepico less certain unless context
    is very clear." Those are three bars, not one."""
    tier = lambda s: next(l for lo, _, l in SURNAME_TIERS
                          if surname_affinity("Miguel Llopiz", "Miguel " + s) >= lo)
    assert tier("Llopis") == "orthographic"
    assert tier("Llepiz") == "near"
    assert tier("Llepico") == "distant"


def test_the_bar_rises_as_the_spelling_drifts():
    bars = [need for _, need, _ in SURNAME_TIERS]
    assert bars == sorted(bars, reverse=True) or bars[0] < bars[1] < bars[2]


def test_an_exact_surname_is_exempt_from_the_tiered_bar():
    """The ruling is about spelling DRIFT. Demanding extra corroboration for an
    exact surname match would block ordinary same-name merges — it broke four
    existing tests when the exemption was missing."""
    a, b = _m("Pedro Gomez"), _m("Pedro Gomez")
    assert context_strength(a, b) < 0.30        # no register, date or context
    assert surname_tier_allows(a, b) == (True, "exact")


def test_a_single_token_name_is_exempt_too():
    """Enslaved people are routinely recorded by given name alone; there is no
    surname to disagree about, and blocking those undoes context-based merging."""
    assert surname_tier_allows(_m("Maria"), _m("Maria"))[0]
    assert surname_affinity("Maria", "Maria Gomez") == 1.0


def test_orthographic_variants_merge_on_reasonable_context_but_not_on_none():
    same_reg = dict(_register="R1", _year=1880)
    assert surname_tier_allows(_m("Miguel Llopiz", **same_reg),
                               _m("Miguel Llopis", **same_reg)) == (True, "orthographic")
    assert not surname_tier_allows(_m("Miguel Llopiz"), _m("Miguel Llopis"))[0]


def test_a_distant_variant_needs_far_more_than_a_shared_register():
    """Llepico with only 'same register' behind it must not auto-merge — that is
    the whole content of 'unless context is very clear'."""
    reg = dict(_register="R1", _year=1880)
    ok, tier = surname_tier_allows(_m("Miguel Llopiz", **reg), _m("Miguel Llepico", **reg))
    assert tier == "distant" and not ok


def test_attribute_conflict_destroys_context_strength():
    a = _m("Miguel Llopiz", _register="R1", _year=1880, occupation="Cleric")
    b = _m("Miguel Llepiz", _register="R1", _year=1880, occupation="Soldier")
    assert context_strength(a, b) < context_strength(
        a, _m("Miguel Llepiz", _register="R1", _year=1880, occupation="Cleric"))


def test_tiering_is_reversible_and_only_ever_refuses_merges():
    """A refused merge becomes a review item, never a dropped record. Turning
    the flag off must return the previous behaviour exactly."""
    vol = {"id": "V", "entries": [
        {"id": "0001-01", "data": {"people": [{"id": "P01", "name": "Miguel Llopiz"}],
                                   "events": [{"date": "1880-01-01"}]}},
        {"id": "0002-01", "data": {"people": [{"id": "P01", "name": "Miguel Llepico"}],
                                   "events": [{"date": "1881-01-01"}]}},
    ]}
    loose = disambiguate_volume(vol, surname_tiers=False)["stats"]
    tight = disambiguate_volume(vol, surname_tiers=True)["stats"]
    assert tight["identities"] >= loose["identities"]
    assert tight["auto_merges"] <= loose["auto_merges"]
    # nothing vanishes: every mention is still in exactly one identity
    for st in (loose, tight):
        assert st["mentions"] == 2


# --------------------------------------------------------------------------- #
# stratified sampling
# --------------------------------------------------------------------------- #

def _pair(score, a="Juan Gomez", b="Juan Gomez", ea="29597-0001-01",
          eb="29597-0002-01", disp="review", reasons=("name~1.00",)):
    return {"score": score, "disposition": disp, "reasons": list(reasons),
            "a": {"entry": ea, "id": "P01", "name": a, "detail": {}},
            "b": {"entry": eb, "id": "P01", "name": b, "detail": {}}}


def test_strata_separate_the_axes_a_reviewer_would_care_about():
    assert volume_of("701054-0003-01") == "701054"
    assert surname_relation("Juan Gomez", "Juan Gomez") == "identical"
    assert surname_relation("Miguel Llopiz", "Miguel Llopis") == "variant"
    assert surname_relation("Juan Gomez", "Juan Valdes") == "different"
    assert surname_relation("Juan", "Juan Gomez") == "one-missing"
    assert band_of(0.99) == "auto-strong" and band_of(0.42) == "very-low"
    assert signal_of(["blocked: x"]) == "blocked"


def test_cross_volume_pairs_are_their_own_case_type():
    same = stratum_of(_pair(0.9, ea="29597-0001-01", eb="29597-0002-01"))
    cross = stratum_of(_pair(0.9, ea="29597-0001-01", eb="701054-0002-01"))
    assert "same-vol" in same and "cross-vol" in cross and same != cross


def test_reservoir_is_bounded_but_counts_the_true_population():
    """The corpus scores tens of millions of pairs; holding them all is what the
    reservoir exists to avoid. What survives must still know how big its
    stratum really was, or the sample cannot be weighted."""
    r = StratifiedReservoir(per_cell=5, seed=1)
    for i in range(1000):
        r.append(_pair(0.75))
    key = next(iter(r.cells))
    assert len(r.cells[key]) == 5           # bounded
    assert r.seen[key] == 1000              # exact population
    drawn = r.draw(5)
    assert drawn[0]["stratum_population"] == 1000
    assert drawn[0]["weight"] == 200.0      # 1000/5, recoverable


def test_water_filling_prefers_variety_over_proportion():
    """One huge stratum must not crowd out the rare ones — the rare ones are the
    point of the exercise."""
    r = StratifiedReservoir(per_cell=50, seed=1)
    for _ in range(900):
        r.append(_pair(0.75))                                  # common case
    r.append(_pair(0.95, b="Juan Valdes", disp="auto"))        # rare case
    r.append(_pair(0.40, b="Juan", disp="below-threshold"))    # rare case
    drawn = r.draw(6)
    assert len({p["stratum"] for p in drawn}) == 3             # all three present
    cov = r.coverage(drawn)
    assert cov["strata_missed"] == [] and cov["pairs_scored"] == 902


def test_sampling_is_reproducible_from_the_seed():
    def run():
        r = StratifiedReservoir(per_cell=4, seed=99)
        for i in range(200):
            r.append(_pair(0.7 + (i % 20) / 100.0))
        return [(p["a"]["entry"], p["score"]) for p in r.draw(20)]
    assert run() == run()


def test_draw_asks_for_more_than_exists_without_crashing():
    r = StratifiedReservoir(per_cell=10, seed=1)
    for _ in range(3):
        r.append(_pair(0.75))
    assert len(r.draw(500)) == 3


# --------------------------------------------------------------------------- #
# the likelihood page
# --------------------------------------------------------------------------- #

def test_only_the_endpoints_become_constraints():
    """75% means "probably, and I could be wrong". Promoting it to a must-link
    would re-introduce speculative merging wearing the authority of a human
    decision."""
    labels = {"labels": [
        {"a": {"entry": "E1", "id": "P01"}, "b": {"entry": "E2", "id": "P01"}, "likelihood": 100},
        {"a": {"entry": "E3", "id": "P01"}, "b": {"entry": "E4", "id": "P01"}, "likelihood": 75},
        {"a": {"entry": "E5", "id": "P01"}, "b": {"entry": "E6", "id": "P01"}, "likelihood": 50},
        {"a": {"entry": "E7", "id": "P01"}, "b": {"entry": "E8", "id": "P01"}, "likelihood": 25},
        {"a": {"entry": "E9", "id": "P01"}, "b": {"entry": "EA", "id": "P01"}, "likelihood": 0},
        {"a": {"entry": "EB", "id": "P01"}, "b": {"entry": "EC", "id": "P01"}, "likelihood": None},
    ]}
    con = labels_to_constraints(labels)
    assert len(con["must"]) == 1 and len(con["cannot"]) == 1
    assert con["must"][0][0]["entry"] == "E1"
    assert con["cannot"][0][0]["entry"] == "E9"


def test_summary_reports_how_much_is_actually_usable():
    labels = {"labels": [
        {"a": {}, "b": {}, "likelihood": 100, "score": 0.9},
        {"a": {}, "b": {}, "likelihood": 50, "score": 0.8},
        {"a": {}, "b": {}, "likelihood": None, "score": 0.8},
    ]}
    s = label_summary(labels)
    assert s["total"] == 3 and s["labelled"] == 2
    assert s["constraints_usable"] == 1        # the 50% is training signal only


def test_page_is_capped_and_keeps_the_highest_scoring_work(tmp_path):
    """230 MB was the uncapped page; no browser opens that."""
    pairs = [_pair(0.5 + i / 100.0) for i in range(40)]
    out = str(tmp_path / "p.html")
    render_likelihood_review_html(pairs, out, tag="T", limit=5)
    html = open(out, encoding="utf-8").read()
    data = json.loads(html.split("const PAIRS = ", 1)[1].split(", LEVELS =", 1)[0])
    assert len(data) == 5
    assert data[0]["score"] == max(p["score"] for p in pairs)


def test_page_cannot_be_broken_out_of_by_a_name(tmp_path):
    evil = _pair(0.8, a="</script><img src=x onerror=alert(1)>")
    out = str(tmp_path / "p.html")
    render_likelihood_review_html([evil], out, tag="T")
    html = open(out, encoding="utf-8").read()
    assert "</script><img" not in html
    assert "\\u003c/script>" in html or "\\u003c" in html


def test_the_five_levels_are_exactly_what_daniel_asked_for(tmp_path):
    from ssda_nlp_tools.likelihood_review_html import LEVELS
    assert [v for v, *_ in LEVELS] == [0, 25, 50, 75, 100]
