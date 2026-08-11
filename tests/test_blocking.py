"""Blocking must not lose a pair the old scan would have kept.

`_shares_context` keeps a pair when ANY of these hold: same register, a person
named in both entries, either side undated, or the two dated within 60 years.
Blocking replaces that after-the-fact filter with keys, so each of those
conditions needs a key that puts the pair together -- and the tests below are
one per condition, because a missing one is silent.

That is not hypothetical: the first version gave undated mentions their own key
without giving it to dated ones, so an undated mention met nobody, and 4,034,031
candidate pairs vanished without any error.
"""
import collections

import ssda_nlp_tools.blocking as B


def _m(name, year=None, register="V1", entry="V1-1", ctx=()):
    return {"name": name, "_year": year, "_register": register,
            "_entry": entry, "_ctx": set(ctx)}


def _pairs(mentions, max_block=B.DEFAULT_MAX_BLOCK):
    return {(i, j) for i, j in B.candidate_pairs(mentions, max_block)}


def test_same_register_is_a_candidate():
    ms = [_m("Maria", 1700, register="V1", entry="V1-1"),
          _m("Maria", 1900, register="V1", entry="V1-2")]
    assert _pairs(ms) == {(0, 1)}, "same register, however far apart in time"


def test_a_shared_associate_is_a_candidate_across_registers():
    """The most valuable output of the whole stage is a cross-register link."""
    ms = [_m("Maria", 1700, register="V1", entry="V1-1",
             ctx=[("parent", "ana lopez")]),
          _m("Maria", 1900, register="V2", entry="V2-1",
             ctx=[("godparent", "ana lopez")])]
    assert _pairs(ms) == {(0, 1)}


def test_close_in_time_is_a_candidate_across_registers():
    ms = [_m("Maria", 1800, register="V1", entry="V1-1"),
          _m("Maria", 1840, register="V2", entry="V2-1")]
    assert _pairs(ms) == {(0, 1)}


def test_every_gap_within_the_window_shares_a_time_key():
    """Exhaustive over the boundary rather than a couple of spot checks --
    an off-by-one in the bucket arithmetic loses real pairs silently."""
    for base in (0, 1, 59, 60, 61, 119, 120, 1799, 1800):
        for gap in range(0, B.YEAR_WINDOW + 1):
            ka = set(B._time_keys(base))
            kb = set(B._time_keys(base + gap))
            assert ka & kb, f"gap {gap} at base {base} shares no time key"


def test_far_apart_in_time_is_not_a_candidate_across_registers():
    ms = [_m("Maria", 1700, register="V1", entry="V1-1"),
          _m("Maria", 1900, register="V2", entry="V2-1")]
    assert _pairs(ms) == set(), "200 years apart, different registers, nobody shared"


def test_an_undated_mention_meets_its_namesakes():
    """The bug that cost 4,034,031 pairs. `_shares_context` keeps a pair when
    EITHER side lacks a year, so an undated mention is a candidate against every
    namesake -- which a shared key cannot express, because dated mentions would
    have to carry it too."""
    ms = [_m("Maria", None, register="V1", entry="V1-1"),
          _m("Maria", 1900, register="V2", entry="V2-1"),
          _m("Maria", 1500, register="V3", entry="V3-1")]
    got = _pairs(ms)
    assert (0, 1) in got and (0, 2) in got
    assert (1, 2) not in got, "two DATED mentions 400 years apart still must not pair"


def test_each_pair_is_yielded_exactly_once():
    """A pair matching several keys must not be scored several times."""
    ms = [_m("Maria", 1800, register="V1", entry="V1-1",
             ctx=[("parent", "ana"), ("godparent", "juan")]),
          _m("Maria", 1801, register="V1", entry="V1-2",
             ctx=[("parent", "ana"), ("godparent", "juan")])]
    got = list(B.candidate_pairs(ms))
    assert got == [(0, 1)], f"expected one pair, got {got}"


def test_a_different_name_is_never_a_candidate():
    ms = [_m("Maria", 1800, register="V1", entry="V1-1"),
          _m("Rodrigo", 1800, register="V1", entry="V1-2")]
    assert _pairs(ms) == set()


def test_a_common_name_loses_only_its_time_key():
    """The bound that makes this scale. Above max_block a name may no longer be
    paired on name-and-date alone -- Daniel: "In the absence of literally any
    other context, records should never be merged" -- but register and shared
    associates still apply, because those carry real evidence."""
    ms = [_m("Maria", 1800 + i, register="V1", entry=f"V1-{i}") for i in range(6)]
    ms.append(_m("Maria", 1802, register="V2", entry="V2-1"))
    ms.append(_m("Maria", 1803, register="V3", entry="V3-1",
                 ctx=[("parent", "ana lopez")]))
    ms.append(_m("Maria", 1804, register="V4", entry="V4-1",
                 ctx=[("parent", "ana lopez")]))
    got = _pairs(ms, max_block=3)          # forces the name over the cap
    # cross-register, name+date only -> dropped
    assert (0, 6) not in got
    # cross-register but sharing an associate -> kept
    assert (7, 8) in got
    # same register -> kept
    assert (0, 1) in got


def test_block_size_report_agrees_with_what_is_generated():
    ms = [_m("Maria", 1800 + i, register="V1", entry=f"V1-{i}") for i in range(10)]
    rep = B.block_size_report(ms)
    assert rep["largest"] >= 2
    # the report is an upper bound: it counts before cross-key de-duplication
    assert rep["pairs_upper_bound"] >= len(_pairs(ms))
