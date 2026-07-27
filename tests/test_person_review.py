"""Per-person review surface: it must REGROUP the queue, never edit it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools.person_review import group_by_person, summarize  # noqa: E402


def _pair(ae, ai, an, be, bi, bn, score):
    return {"score": score, "reasons": [f"name~{score}"],
            "a": {"entry": ae, "id": ai, "name": an, "detail": {}},
            "b": {"entry": be, "id": bi, "name": bn, "detail": {}}}


QUEUE = [
    _pair("E1", "P01", "Maria", "E2", "P01", "Maria", 0.81),
    _pair("E1", "P01", "Maria", "E3", "P02", "Maria", 0.75),
    _pair("E4", "P01", "Juan", "E5", "P03", "Juan", 0.72),
]


def test_every_pair_survives_under_both_people():
    """The regrouping must not drop a candidate. A reviewer may arrive at the
    decision from either side, so each pair appears under both of its people."""
    screens = group_by_person(QUEUE)
    rows = sum(s["n_candidates"] for s in screens)
    assert rows == 2 * len(QUEUE)          # nothing lost, each pair seen twice
    maria_e1 = next(s for s in screens
                    if s["person"]["entry"] == "E1" and s["person"]["id"] == "P01")
    assert maria_e1["n_candidates"] == 2   # its two candidates on ONE screen
    assert maria_e1["best_score"] == 0.81


def test_screens_are_ordered_by_strongest_candidate():
    screens = group_by_person(QUEUE)
    assert [s["best_score"] for s in screens] == sorted(
        (s["best_score"] for s in screens), reverse=True)
    assert screens[0]["candidates"][0]["score"] >= screens[0]["candidates"][-1]["score"]


def test_min_score_filters_without_reordering():
    screens = group_by_person(QUEUE, min_score=0.80)
    assert sum(s["n_candidates"] for s in screens) == 2   # only the 0.81 pair, both sides
    assert all(c["score"] >= 0.80 for s in screens for c in s["candidates"])


def test_summary_does_not_invent_people_with_no_candidates():
    """Identities with no candidate never enter the review queue, so they cannot
    be counted from it — the total must be supplied, not inferred."""
    screens = group_by_person(QUEUE)
    rep = summarize(screens, total_identities=1000, total_pairs=len(QUEUE))
    assert rep["screens_with_candidates"] == len(screens)
    assert rep["identities_needing_no_review"] == 1000 - len(screens)
    assert rep["pair_queue_length"] == 3
    # and with no total supplied it stays silent rather than guessing
    bare = summarize(screens)
    assert "total_identities" not in bare
    assert "identities_needing_no_review" not in bare


def test_empty_queue_is_not_an_error():
    assert group_by_person([]) == []
    rep = summarize([], total_identities=42)
    assert rep["screens_with_candidates"] == 0
    assert rep["identities_needing_no_review"] == 42


def test_identity_map_refuses_to_build_from_id_less_records():
    """The first version looked for 'global_id'/'id' but person_index.json uses
    'person_id'. Every lookup returned None, so the whole corpus collapsed onto
    one screen and every candidate was discarded as a self-match. Fail loudly."""
    import pytest
    from ssda_nlp_tools.person_review import mention_to_identity
    real = [{"person_id": "V-0001", "mentions": [{"entry": "E1", "id": "P01"}]}]
    assert mention_to_identity(real) == {"E1::P01": "V-0001"}
    with pytest.raises(ValueError):
        mention_to_identity([{"mentions": [{"entry": "E1", "id": "P01"}]}])


def test_grouping_by_identity_collapses_a_person_seen_in_many_entries():
    """Grouping by MENTION is not per-person: someone in three entries yields
    three screens. With the identity map it is one."""
    from ssda_nlp_tools.person_review import group_by_person
    q = [_pair("E1", "P01", "Ana", "E9", "P01", "Beatriz", 0.75),
         _pair("E2", "P01", "Ana", "E9", "P01", "Beatriz", 0.74),
         _pair("E3", "P01", "Ana", "E9", "P01", "Beatriz", 0.73)]
    assert len(group_by_person(q)) == 4                    # 3 mentions + 1 other
    m2i = {"E1::P01": "ANA", "E2::P01": "ANA", "E3::P01": "ANA", "E9::P01": "BEA"}
    screens = group_by_person(q, identity_of=m2i)
    assert len(screens) == 2                               # one per real person
    ana = next(s for s in screens if s["person"]["identity"] == "ANA")
    assert ana["n_candidates"] == 1                        # Beatriz listed once, not 3x
