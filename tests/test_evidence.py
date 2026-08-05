"""Weight-of-evidence scoring: the constraints Daniel stated, as tests."""
import math

import pytest

from ssda_nlp_tools.evidence import (AUTO_MERGE_LOG_ODDS, LOG_PRIOR_ODDS,
                                     MAX_NAME_LLR, REVIEW_LOG_ODDS, NameStats,
                                     network_llr, score)


def _stats(names):
    return NameStats([{"name": n} for n in names])


def test_a_name_alone_can_never_auto_merge():
    """Daniel, 2026-07-29: "No people should be merged strictly based on name
    correspondence." That is a constraint on MAX_NAME_LLR, and the first version
    violated it: 9.0 - 5.56 = +3.44 against a +3.00 threshold."""
    assert LOG_PRIOR_ODDS + MAX_NAME_LLR < AUTO_MERGE_LOG_ODDS
    s = _stats(["ana solar"] * 3)
    r = score({"name": "Zoraida Quintanilla", "_entry": "E1"},
              {"name": "Zoraida Quintanilla", "_entry": "E2"}, s)
    assert r["decision"] != "merge"


def test_a_bare_name_match_is_dropped_rather_than_queued():
    """A DELIBERATE TRADE, recorded because it was not free.

    Two constraints came into conflict once the circumstantial weights were
    measured rather than guessed:

      (a) name + ALL non-discriminative evidence must not auto-merge
      (b) a matching name should stay a live candidate for review

    Satisfying (a) with the measured weights (same-city +1.45, close-date +0.27)
    forces MAX_NAME_LLR down to 5.5, which puts a bare name match at -0.06 --
    a hair BELOW the review line, so (b) is given up.

    That is the right way round. Daniel's ruling is about merging, not about
    reviewing; and queueing every bare name match in a corpus this full of
    Marias is precisely the 10%+ review rate he called unacceptable. The cost is
    real: a genuine match evidenced by nothing but a name will be dropped
    silently rather than shown to a human.
    """
    assert LOG_PRIOR_ODDS + MAX_NAME_LLR < REVIEW_LOG_ODDS
    assert LOG_PRIOR_ODDS + MAX_NAME_LLR > -1.0, "and only just below"


def test_rare_names_carry_more_evidence_than_common_ones():
    s = _stats(["maria"] * 500 + ["custodio vieira"])
    assert s.llr("custodio vieira") > s.llr("maria")


def test_sharing_associates_is_positive_evidence():
    s = _stats(["x"] * 100)
    llr, why = network_llr(
        {"relations": [{"type": "parent", "name": "Ines Bacallao"}]},
        {"relations": [{"type": "parent", "name": "Ines Bacallao"}]}, s)
    assert llr > 0 and any("shared" in w for w in why)


def test_dense_disjoint_networks_are_negative_evidence():
    """The case the old scorer got wrong 30 times out of 30: identical names,
    dense networks, nobody in common."""
    s = _stats(["x"] * 100)
    a = {"relations": [{"type": "parent", "name": f"A{i}"} for i in range(4)]}
    b = {"relations": [{"type": "parent", "name": f"B{i}"} for i in range(4)]}
    llr, why = network_llr(a, b, s)
    assert llr < 0 and any("disjoint" in w for w in why)


def test_a_thin_network_is_absence_of_evidence_not_evidence_of_absence():
    """One record with six relatives and one with none must not be penalised;
    only a genuine clash of two dense networks counts against."""
    s = _stats(["x"] * 100)
    a = {"relations": [{"type": "parent", "name": f"A{i}"} for i in range(6)]}
    llr, _ = network_llr(a, {"relations": []}, s)
    assert llr == 0.0


def test_vetoes_are_impossibilities_and_cannot_be_outvoted():
    """No amount of agreement may merge two people from one entry."""
    s = _stats(["x"] * 10)
    r = score({"name": "Ana Solar", "_entry": "E1", "phenotype": "parda"},
              {"name": "Ana Solar", "_entry": "E1", "phenotype": "parda"}, s)
    assert r["vetoed"] == "same-entry" and r["probability"] == 0.0


def test_conflicting_attributes_push_apart():
    s = _stats(["x"] * 10)
    same = score({"name": "Ana Solar", "_entry": "E1", "free": True},
                 {"name": "Ana Solar", "_entry": "E2", "free": True}, s)
    diff = score({"name": "Ana Solar", "_entry": "E1", "free": True},
                 {"name": "Ana Solar", "_entry": "E2", "free": False}, s)
    assert diff["log_odds"] < same["log_odds"]


def test_every_score_is_explainable():
    s = _stats(["x"] * 10)
    r = score({"name": "Ana Solar", "_entry": "E1"},
              {"name": "Ana Solar", "_entry": "E2"}, s)
    assert r["terms"] and all(isinstance(w, float) for _, w in r["terms"])
    assert abs(r["log_odds"] - (LOG_PRIOR_ODDS + sum(w for _, w in r["terms"]))) < 1e-9


def test_place_weights_are_monotone_in_closeness():
    """Co-location cannot become STRONGER evidence as it gets coarser.

    Fitting the levels independently produced city=+1.45 against
    institution=+0.50 -- same city rated above same parish. It came from a
    single positive pair, because with seven volumes and no two sharing an
    institution a "place level" is a lookup on a VOLUME PAIR, not an independent
    observation, and `city` names exactly one pair (201991/29597). Those two
    registers do not even overlap in time.
    """
    from ssda_nlp_tools.evidence import W_PLACE
    order = ["institution", "city", "state", "country", "none"]
    vals = [W_PLACE[k] for k in order]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), vals


def test_no_place_level_can_carry_a_merge_by_itself():
    """Location is corroboration, never proof. Daniel calls it critical, and it
    is -- but 56.7% of candidate pairs already share an institution."""
    from ssda_nlp_tools.evidence import (LOG_PRIOR_ODDS, W_PLACE,
                                         AUTO_MERGE_LOG_ODDS, MAX_NAME_LLR)
    best = max(W_PLACE.values())
    assert LOG_PRIOR_ODDS + MAX_NAME_LLR + best < AUTO_MERGE_LOG_ODDS
