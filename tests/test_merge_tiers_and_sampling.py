"""Daniel's 2026-07-29 answers, as executable rules.

Covers the surname tiering (his Llopiz ruling), the stratified training sample,
and the 0/25/50/75/100 labelling page.
"""
import json
import re

import pytest

from ssda_nlp_tools.disambiguate import (MIN_SIGNALS_FOR_ANY_MERGE,
                                         SURNAME_TIERS, context_strength,
                                         corroborating_signals,
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
    bar = lambda s: next(need for lo, need, _ in SURNAME_TIERS
                         if surname_affinity("Miguel Llopiz", "Miguel " + s) >= lo)
    # three cases, three increasing evidential bars -- his sentence as numbers
    assert bar("Llopis") < bar("Llepiz") < bar("Llepico")
    assert bar("Llopis") == MIN_SIGNALS_FOR_ANY_MERGE


def test_the_bar_rises_as_the_spelling_drifts():
    bars = [need for _, need, _ in SURNAME_TIERS]
    assert bars == sorted(bars), bars           # monotonic, exact -> distant
    assert bars[0] >= MIN_SIGNALS_FOR_ANY_MERGE  # even an exact match needs some


def test_even_an_exact_surname_needs_corroboration():
    """Daniel, 2026-07-29: "No people should be merged strictly based on name
    correspondence; it should depend on a combination of date overlap,
    same-named relation, same/similar qualities."

    This test asserted the opposite for two days. The exemption it defended was
    the root of both defects found since: the devotional-epithet merges and the
    41 women recorded only as "Maria" were each special cases of a rule that let
    a name alone carry a merge."""
    a, b = _m("Pedro Gomez"), _m("Pedro Gomez")
    assert len(corroborating_signals(a, b)) < MIN_SIGNALS_FOR_ANY_MERGE
    assert surname_tier_allows(a, b) == (False, "exact")
    # with a date overlap and a same-named relation, it merges
    ctx = dict(_year=1880, _ctx={("spouse", "ana ruiz")}, occupation="soldier")
    assert surname_tier_allows(_m("Pedro Gomez", **ctx),
                               _m("Pedro Gomez", **ctx)) == (True, "exact")


def test_a_bare_given_name_needs_a_person_named_in_both_entries():
    """This test previously asserted the opposite, that a surname-less name was
    exempt. Measurement overturned it: the exemption merged 41 women recorded
    only as "Maria" into a single identity across two registers, plus 36
    "Francisco" and 21 "Rafael".

    Absence of a surname is absence of evidence, not permission. The bar is
    specifically a shared third party, because sharing a register and a rough
    date is what every bare "Maria" in a parish has in common -- that is what
    let them chain -- whereas a shared enslaver or spouse is how these registers
    actually identify someone with no surname."""
    assert not surname_tier_allows(_m("Maria", _register="R1", _year=1880),
                                   _m("Maria", _register="R1", _year=1881))[0]
    linked = dict(_register="R1", _year=1880, _ctx={("enslaver", "juan vives")})
    assert surname_tier_allows(_m("Maria", **linked), _m("Maria", **linked)) ==         (True, "uninformative")
    assert surname_affinity("Maria", "Maria Gomez") == 1.0


def test_nomen_nescio_placeholders_are_not_surnames():
    """'Francisco N.' is Francisco surname-unknown. Treating N. as a surname let
    it match another 'Francisco N.' EXACTLY, so two people merged on the
    strength of both being unnamed."""
    from ssda_nlp_tools.disambiguate import is_placeholder_surname, _surname_of
    assert _surname_of("Francisco N.") == "n"
    assert is_placeholder_surname("n")          # nomen nescio
    assert is_placeholder_surname("C")          # bare initial
    assert is_placeholder_surname("desconocido")
    assert is_placeholder_surname(None)         # no surname at all
    assert not is_placeholder_surname("gomez")


def test_canonical_name_is_the_form_actually_used_most():
    """"Longest wins" promoted a transcription artefact over the real name: 31
    'Maria' plus one stray 'Maria Maria' was labelled 'Maria Maria'."""
    from ssda_nlp_tools.disambiguate import disambiguate_volume
    people = [{"id": "P01", "name": "Maria"}, {"id": "P02", "name": "Juan Vives"}]
    entries = [{"id": f"000{i}-01", "data": {"people": [
        {**people[0]}, {**people[1]},
    ], "events": [{"type": "baptism", "date": f"188{i}-01-01",
                   "principals": ["P02"]}]}} for i in range(1, 4)]
    entries[2]["data"]["people"][0]["name"] = "Maria Maria"
    for e in entries:
        e["data"]["people"][0]["relationships"] = [
            {"related_person": "P02", "relationship_type": "enslaver"}]
    res = disambiguate_volume({"id": "V", "entries": entries})
    marias = [i for i in res["identities"] if "aria" in (i["canonical_name"] or "")]
    biggest = max(marias, key=lambda i: i["n_mentions"])
    assert biggest["canonical_name"] == "Maria"      # modal, not longest


def test_orthographic_variants_merge_on_reasonable_context_but_not_on_none():
    """Llopiz and Llopis fold to the same phonetic form, so they now land in the
    `exact` tier rather than `orthographic`. The tier LABEL is incidental; both
    carry the same bar, and the bar is what Daniel's ruling was about."""
    # a date overlap AND agreeing qualities: two signals, so it merges
    corroborated = dict(_register="R1", _year=1880, occupation="Cleric")
    assert surname_tier_allows(_m("Miguel Llopiz", **corroborated),
                               _m("Miguel Llopis", **corroborated))[0]
    # the same pair on a shared register and a date alone is ONE signal, and a
    # shared register is not a signal at all -- everyone in a volume has it
    thin = dict(_register="R1", _year=1880)
    assert not surname_tier_allows(_m("Miguel Llopiz", **thin),
                                   _m("Miguel Llopis", **thin))[0]
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


# --------------------------------------------------------------------------- #
# devotional epithets in the surname slot
# --------------------------------------------------------------------------- #

def test_devotional_epithets_are_recognised_in_the_surname_slot():
    from ssda_nlp_tools.disambiguate import _surname_of, is_devotional_epithet
    assert _surname_of("María Josefa de la Concepción") == "concepcion"
    assert is_devotional_epithet("concepcion")
    assert is_devotional_epithet("Cruz")            # accent/case insensitive
    assert not is_devotional_epithet("llopiz")
    assert not is_devotional_epithet("gomez")
    assert not is_devotional_epithet(None)


def test_an_exact_epithet_match_no_longer_skips_every_tier():
    """The bug: `_surname_of` reads "de la Cruz" as the surname "Cruz", two
    unrelated women both match it EXACTLY, and an exact match took the exemption
    that bypasses all the tiers. The most-shared names got the least scrutiny.
    63 merged identities and 699 mentions rested on this."""
    bare = dict(_register=None, _year=None)
    ok, _ = surname_tier_allows(_m("María Josefa de la Cruz", **bare),
                                _m("María Josefa de la Cruz", **bare))
    assert not ok                                # no corroboration -> refused


def test_a_real_family_still_merges_on_corroboration():
    """This is a bar, not a ban. A genuine Cruz family shares a register,
    relatives and dates; two strangers sharing a Marian epithet do not."""
    ctx = dict(_register="R1", _year=1880, _ctx={("parent", "juan perez")})
    ok, _ = surname_tier_allows(_m("María de la Cruz", **ctx),
                                _m("María de la Cruz", **ctx))
    assert ok


def test_clergy_in_consecutive_records_are_the_one_sanctioned_shortcut():
    """Daniel's carve-out: "perhaps a very strict rules-based starting point for
    obvious merges like the clergy that appear in many consecutive records"."""
    p = dict(occupation="cleric", _register="201991")
    a = _m("Miguel O'Reilly", _entry="201991-0004-A-01", **p)
    b = _m("Miguel O'Reilly", _entry="201991-0005-A-02", **p)
    assert surname_tier_allows(a, b) == (True, "clergy-consecutive")
    # far apart in the book: no longer "consecutive", so back to normal rules
    far = _m("Miguel O'Reilly", _entry="201991-0299-A-02", **p)
    assert surname_tier_allows(a, far)[1] != "clergy-consecutive"
    # a layman with the same name gets no shortcut
    lay = dict(_register="201991")
    assert surname_tier_allows(_m("Miguel O'Reilly", _entry="201991-0004-A-01", **lay),
                               _m("Miguel O'Reilly", _entry="201991-0005-A-02", **lay)
                               )[1] != "clergy-consecutive"


def test_a_missing_epithet_file_falls_back_to_previous_behaviour(monkeypatch):
    """The list is data, not code. If it goes missing the stage must degrade to
    what it did before, not crash mid-corpus."""
    import ssda_nlp_tools.disambiguate as D
    monkeypatch.setattr(D, "_EPITHETS_PATH", "does-not-exist.json")
    monkeypatch.setattr(D, "_epithets_cache", None)
    assert D._load_epithets()[0] == set()
    assert not D.is_devotional_epithet("cruz")
    D._epithets_cache = None


# --------------------------------------------------------------------------- #
# reading Daniel's labels back
# --------------------------------------------------------------------------- #

def _lbl(likelihood, disposition, a="E1", b="E2"):
    return {"a": {"entry": a, "id": "P01"}, "b": {"entry": b, "id": "P01"},
            "likelihood": likelihood, "disposition": disposition, "score": 0.9}


def test_a_refused_pair_labelled_same_is_the_rule_being_too_strict():
    """The direction that carries new information. Every measurement so far
    could only reveal over-merging, because that is what shows up in the graph;
    a too-strict rule fails silently by leaving two records apart."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("al", "analyze_labels.py")
    al = importlib.util.module_from_spec(spec); spec.loader.exec_module(al)
    assert al.verdict(_lbl(100, "blocked-surname-tier-exact")) == "too_strict"
    assert al.verdict(_lbl(75, "blocked-uninformative")) == "too_strict"
    assert al.verdict(_lbl(0, "auto")) == "too_loose"
    assert al.verdict(_lbl(100, "auto")) == "agree"
    assert al.verdict(_lbl(0, "blocked-cluster-surname")) == "agree"


def test_fifty_percent_counts_as_neither_agreement_nor_disagreement():
    """50% is the reviewer saying he cannot tell. Scoring it either way would
    manufacture a signal out of an admission of uncertainty."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("al", "analyze_labels.py")
    al = importlib.util.module_from_spec(spec); spec.loader.exec_module(al)
    assert al.verdict(_lbl(50, "auto")) == "unclear"
    assert al.verdict(_lbl(50, "review")) == "unclear"
    assert al.verdict(_lbl(None, "auto")) is None


def test_only_a_plain_auto_counts_as_the_algorithm_having_merged():
    """Every other disposition is a refusal of some kind, including the
    sacrament and cluster guards, which do not carry the word 'blocked' in a
    way worth pattern-matching on."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("al", "analyze_labels.py")
    al = importlib.util.module_from_spec(spec); spec.loader.exec_module(al)
    assert al.merged_by_algorithm("auto")
    for d in ("review", "below-threshold", "blocked-sacrament-principal",
              "blocked-cluster-surname", "blocked-surname-tier-near"):
        assert not al.merged_by_algorithm(d), d


def test_labels_are_keyed_by_pair_identity_not_array_position():
    """The sample was regenerated twice in one day. Under position-keyed
    storage a reviewer who had labelled the earlier build would reopen the new
    one to find it apparently pre-filled, every answer silently attached to a
    different pair."""
    import tempfile, os
    a = [_pair(0.9, a="Ana", ea="E1"), _pair(0.8, a="Bea", ea="E2")]
    b = [_pair(0.8, a="Bea", ea="E2"), _pair(0.9, a="Ana", ea="E1")]   # reordered
    d = tempfile.mkdtemp()
    fa, fb = os.path.join(d, "a.html"), os.path.join(d, "b.html")
    render_likelihood_review_html(a, fa, tag="core")
    render_likelihood_review_html(b, fb, tag="core")
    ha, hb = open(fa, encoding="utf-8").read(), open(fb, encoding="utf-8").read()
    fp = lambda h: re.search(r'FINGERPRINT = "([0-9a-f]+)"', h).group(1)
    assert fp(ha) == fp(hb)          # same pairs, order irrelevant -> resumable
    assert "pairKey" in ha and "labels[pairKey(" in ha


def test_a_different_sample_cannot_inherit_the_previous_labels():
    import tempfile, os
    d = tempfile.mkdtemp()
    one = os.path.join(d, "1.html"); two = os.path.join(d, "2.html")
    render_likelihood_review_html([_pair(0.9, ea="E1")], one, tag="core")
    render_likelihood_review_html([_pair(0.9, ea="E9")], two, tag="core")
    fp = lambda p: re.search(r'FINGERPRINT = "([0-9a-f]+)"',
                             open(p, encoding="utf-8").read()).group(1)
    assert fp(one) != fp(two)        # different pairs -> different storage key
