"""The synthetic corner-case set, and the properties that make it useful.

Daniel asked for artificial records so he can rule on true corner cases instead
of "fairly obvious 0/100s". A generator can fail that brief in ways that look
fine from the outside, so these tests hold the properties that matter.
"""
import collections

from build_synthetic_set import predict
from ssda_nlp_tools.synthetic_pairs import FAMILIES, generate


def test_families_are_evenly_represented():
    pairs = generate(300)
    counts = collections.Counter(p["family"] for p in pairs)
    assert len(counts) == len(FAMILIES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_every_pair_carries_its_question():
    for p in generate(60):
        assert p["question"] and p["question"].endswith("?")
        assert p["expected"] is None, "shipping a guess would anchor the labeller"


def test_ids_are_unique_and_stable():
    a, b = generate(80, seed=7), generate(80, seed=7)
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert len({p["id"] for p in a}) == 80


def test_pairs_that_test_one_variable_share_a_name():
    """A family varying chronology must not also vary the name, or the pair is
    trivially different and tests nothing. This caught two families where the
    two sides were given independent random names."""
    for fam in ("lifespan_edge", "shared_given", "temporal_gap",
                "attribute_drift", "clergy_recurrence"):
        for p in [x for x in generate(300) if x["family"] == fam]:
            assert p["a"]["name"] == p["b"]["name"], f"{fam} varies the name too"


def test_the_set_is_not_all_one_answer():
    """A corner-case set our algorithm answers uniformly is a set of obvious
    cases wearing a disguise."""
    preds = [predict(p) for p in generate(300)]
    merges = sum(1 for x in preds if x["would_merge"])
    assert 0.25 < merges / len(preds) < 0.75


def test_most_families_are_individually_mixed():
    """Some uniformity is honest -- we merge every lifespan_edge regardless of
    implied age, which is exactly the gap Daniel should correct. But a set where
    MOST families are uniform is not probing boundaries."""
    preds = collections.defaultdict(list)
    for p in generate(300):
        preds[p["family"]].append(predict(p)["would_merge"])
    mixed = sum(1 for v in preds.values() if 0 < sum(v) < len(v))
    assert mixed >= len(preds) - 3


def test_shared_given_no_longer_merges():
    """Two enslavers sharing a common given name but differing in surname was a
    confirmed false merge in the live data."""
    preds = [predict(p) for p in generate(300) if p["family"] == "shared_given"]
    assert not any(x["would_merge"] for x in preds)
