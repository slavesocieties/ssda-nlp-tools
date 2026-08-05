"""The targeted set must probe the region the existing labels miss."""
import collections

import pytest

from build_targeted_labels import stratum


def _d(rarity=7.0, shared=0, na=0, nb=0, gap=3):
    return {"rarity": rarity, "shared": shared, "na": na, "nb": nb, "gap": gap}


CUTS = [6.48, 7.13, 7.86]


def test_every_axis_is_encoded_independently():
    s = stratum(_d(rarity=6.0, shared=0, na=0, nb=0, gap=2), CUTS)
    assert s == "q1|none|both-thin|<=5y"
    s = stratum(_d(rarity=9.0, shared=2, na=4, nb=5, gap=50), CUTS)
    assert s == "q4|shared|both-dense|>40y"


def test_one_sided_density_is_distinct_from_both_dense():
    """A record with six relatives against one with none is absence of
    evidence; six against six different ones is a clash. They must not share
    a stratum."""
    assert "one-sided" in stratum(_d(na=6, nb=1), CUTS)
    assert "both-dense" in stratum(_d(na=6, nb=6), CUTS)


def test_missing_dates_get_their_own_bucket_not_a_guess():
    assert stratum(_d(gap=None), CUTS).endswith("nodate")


def test_rarity_uses_the_uncapped_value():
    """MAX_NAME_LLR caps at 5.5 and EVERY pair in this region is above it, so
    stratifying on the capped value would put all 29,609 in one bucket."""
    from ssda_nlp_tools.evidence import MAX_NAME_LLR
    assert all(c > MAX_NAME_LLR for c in CUTS)
    assert stratum(_d(rarity=6.1), CUTS) != stratum(_d(rarity=10.0), CUTS)


def test_the_page_never_shows_the_model_answer():
    """Anchoring would measure agreement with us rather than his judgement."""
    import json
    import os
    p = "production/luna_v3/targeted/targeted_pairs.html"
    if not os.path.exists(p):
        pytest.skip("set not built")
    h = open(p, encoding="utf-8").read()
    assert "probability" not in h and "log_odds" not in h
    assert "decision" not in h
