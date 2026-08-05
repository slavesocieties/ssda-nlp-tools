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
        if p["family"].startswith("network"):
            continue          # see test_the_scorer_is_blind_to_social_networks
        preds[p["family"]].append(predict(p)["would_merge"])
    mixed = sum(1 for v in preds.values() if 0 < sum(v) < len(v))
    assert mixed >= len(preds) - 3


def test_the_scorer_is_blind_to_social_networks():
    """A RECORD OF A KNOWN GAP, not a passing feature.

    Daniel, 2026-08-05: people "appear embedded in a social network of some
    density. This is also critical to disambiguation."

    The current scorer merges every one of the four network families, 30 of 30
    each -- including network_conflict, where identical names carry DISJOINT
    networks of 3 to 5 named associates. Two different people who share a name
    and share nobody are merged, because the name matches and the conflicting
    social evidence is never weighed at all.

    This test asserts the CURRENT broken behaviour so the fix is visible when it
    lands: when the probabilistic scorer starts weighing network overlap, this
    test must be updated, and its failure is the signal that it worked.
    """
    by = collections.defaultdict(list)
    for p in generate(420, seed=20260805):
        if p["family"].startswith("network"):
            by[p["family"]].append(predict(p)["would_merge"])
    assert by, "network families missing from the generator"
    for fam, v in by.items():
        assert sum(v) == len(v), (
            f"{fam} is no longer uniformly merged -- if the scorer now weighs "
            f"network evidence, update this test; that is the point of it")


def test_shared_given_no_longer_merges():
    """Two enslavers sharing a common given name but differing in surname was a
    confirmed false merge in the live data."""
    preds = [predict(p) for p in generate(300) if p["family"] == "shared_given"]
    assert not any(x["would_merge"] for x in preds)


# --- social networks (Daniel, 2026-08-05) -----------------------------------

def _net(family, n=40):
    from ssda_nlp_tools.synthetic_pairs import generate
    return [p for p in generate(n=n * 14, seed=7) if p["family"] == family]


def test_network_overlap_actually_varies_size_and_overlap():
    """The point of the family is the SPREAD. A generator that emitted one
    shape would look fine in the output and teach nothing."""
    ps = _net("network_overlap")
    sizes = {len(p["a"]["relations"]) for p in ps}
    shares = {len({r["name"] for r in p["a"]["relations"]} &
                  {r["name"] for r in p["b"]["relations"]}) for p in ps}
    assert len(sizes) >= 3, f"only sizes {sizes}"
    assert len(shares) >= 3, f"only overlaps {shares}"
    assert 0 in shares, "no zero-overlap case"


def test_network_conflict_is_identical_names_with_disjoint_networks():
    """The case a name threshold cannot get right: every string signal says
    merge, the social evidence says two people."""
    for p in _net("network_conflict"):
        assert p["a"]["name"] == p["b"]["name"]
        assert p["a"]["relations"] and p["b"]["relations"]
        assert not ({r["name"] for r in p["a"]["relations"]} &
                    {r["name"] for r in p["b"]["relations"]})


def test_network_asymmetric_is_genuinely_lopsided():
    for p in _net("network_asymmetric"):
        assert len(p["a"]["relations"]) >= 4
        assert len(p["b"]["relations"]) <= 1


def test_associates_within_one_side_are_distinct():
    """Duplicate associates would inflate apparent overlap, which is exactly
    the quantity these families measure."""
    for fam in ("network_overlap", "network_conflict", "network_asymmetric"):
        for p in _net(fam):
            for side in ("a", "b"):
                names = [r["name"] for r in p[side]["relations"]]
                assert len(names) == len(set(names)), f"{fam} {p['id']} {side}"


def test_role_shift_keeps_the_associate_and_changes_the_role():
    for p in _net("network_role_shift"):
        shared = ({r["name"] for r in p["a"]["relations"]} &
                  {r["name"] for r in p["b"]["relations"]})
        assert shared, "role-shift must share the associate"
