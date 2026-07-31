"""SSDA's own hand-labelled merge decisions, as a regression suite.

`disambiguate.json` in slavesocieties/openai holds 59 pairs a person labelled
match true/false. The four NON-matches are the valuable part and they encode
Daniel's ruling directly: every one is a same-name pair judged to be different
people. "No people should be merged strictly based on name correspondence."

The 55 positives are deliberately NOT asserted on. Those gold records carry no
events and therefore no dates, and date overlap is one of the four corroborating
signals a merge needs, so the scorer under-merges on this data structurally.
Measured on our real corpus, 97.2% of 27,875 mentions DO carry a year, so the
gap is a property of the gold file rather than of the pipeline -- which is worth
knowing precisely because a 0/32 recall on gold looks alarming until you check.
"""
import json
import os

import pytest

from run_gold_merge import to_mention
from ssda_nlp_tools.disambiguate import (MIN_SIGNALS_FOR_ANY_MERGE,
                                         corroborating_signals, pair_score,
                                         surname_tier_allows)
from ssda_nlp_tools.textmatch import normalize_name

GOLD = os.path.join(os.path.dirname(__file__), "..", "..",
                    "ssda-openai", "disambiguate.json")


def _would_merge(a, b):
    score, _ = pair_score(a, b)
    allowed, _ = surname_tier_allows(a, b)
    return (allowed and score >= 0.86
            and len(corroborating_signals(a, b)) >= MIN_SIGNALS_FOR_ANY_MERGE)


def _load():
    if not os.path.exists(GOLD):
        pytest.skip("slavesocieties/openai not checked out beside this repo")
    return json.load(open(GOLD, encoding="utf-8"))["manual"]


def test_no_hand_labelled_non_match_is_ever_merged():
    """The precision test. A failure here is a real bug, not a tuning question."""
    gold = _load()
    names = {str(p["id"]): normalize_name(p.get("name"))
             for pair in gold for p in pair["people"]
             if not isinstance(p.get("id"), list)}
    merged = []
    for pair in gold:
        if pair.get("match"):
            continue
        a, b = (to_mention(p, names, "166470") for p in pair["people"])
        if _would_merge(a, b):
            merged.append([p.get("name") for p in pair["people"]])
    assert merged == [], f"merged pairs a person labelled different: {merged}"


def test_the_negatives_are_all_same_name_pairs():
    """If this ever fails the suite has stopped testing what it claims to: the
    whole point is that these are name matches and name is not evidence."""
    gold = _load()
    for pair in gold:
        if pair.get("match"):
            continue
        a, b = (normalize_name(p.get("name")) for p in pair["people"])
        assert a == b, f"expected a same-name negative, got {a!r} vs {b!r}"


def test_identical_name_and_qualities_alone_do_not_merge():
    """The Tomas Angel Joseph case, reduced. Same full name, nothing else."""
    mk = lambda pid: to_mention({"id": pid, "name": "Tomas Angel Joseph"}, {}, "1")
    assert not _would_merge(mk("0044-05-P05"), mk("0044-05-P03"))


def test_one_corroborating_signal_is_never_enough():
    a = to_mention({"id": "0001-01-P01", "name": "Juana Rodriguez",
                    "free": "libre"}, {}, "1")
    b = to_mention({"id": "0002-01-P01", "name": "Juana Rodriguez",
                    "free": "libre"}, {}, "1")
    assert len(corroborating_signals(a, b)) < MIN_SIGNALS_FOR_ANY_MERGE
    assert not _would_merge(a, b)
