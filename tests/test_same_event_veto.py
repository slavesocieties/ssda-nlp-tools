"""The same-event veto must stop co-participants WITHOUT stopping true merges.

One sacramental event can reach the corpus as two entries. Their co-participants
-- siblings sharing parents, two witnesses to one marriage -- then meet as
ordinary cross-entry candidates with every circumstantial term agreeing, and 24%
of the scorable ones auto-merge. Measured in
eval_data/latent_coparticipant_20260813.md.

The trap this file exists to pin: the obvious fix, vetoing every pair across a
shared event, is WRONG. Across two copies of one record, local id P01 and P01 are
the same person and must still merge -- 170 of them do on this corpus. A blanket
event veto destroys those to prevent 50 false ones. The rule is therefore
event-plus-DIFFERENT-local-id.
"""
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.disambiguate import assign_event_ids
from ssda_nlp_tools.evidence import NameStats, score


def _m(entry, lid, name, rels=None, event=None, year=1850):
    m = {"_entry": entry, "_local_id": lid, "name": name,
         "relationships": rels or [], "_ctx": set(), "_year": year,
         "_sacraments": {"baptism"}, "_register": entry.split("-")[0]}
    if event:
        m["_event"] = event
    return m


def _stats(ms):
    return NameStats(ms, is_clergy=lambda n: False)


def test_same_entry_veto_is_unchanged():
    a, b = _m("E1", "P01", "Maria"), _m("E1", "P02", "Juan")
    r = score(a, b, _stats([a, b]))
    assert r["vetoed"] == "same-entry"


def test_different_people_across_one_event_are_vetoed():
    a = _m("E1", "P01", "Paula Corrales", event="EV:E1")
    b = _m("E2", "P02", "Maria Corrales", event="EV:E1")
    r = score(a, b, _stats([a, b]))
    assert r["vetoed"] == "same-event"


def test_the_SAME_person_across_one_event_is_NOT_vetoed():
    """The case a blanket event veto would destroy."""
    a = _m("E1", "P01", "Paula Corrales", event="EV:E1")
    b = _m("E2", "P01", "Paula Corrales", event="EV:E1")
    r = score(a, b, _stats([a, b]))
    assert not r["vetoed"], "the same person across two copies must still merge"


# The names below are deliberately IDENTICAL. An earlier draft used "Paula" vs
# "Maria" and the pre-existing `different-names` veto fired first, so the test
# passed for a reason that had nothing to do with events. Same names isolate the
# event logic as the only thing under test.

def test_absent_event_id_changes_nothing():
    """Backward compatibility: no _event means exactly the old behaviour."""
    a, b = _m("E1", "P01", "Paula Corrales"), _m("E2", "P02", "Paula Corrales")
    assert not score(a, b, _stats([a, b]))["vetoed"]


def test_different_events_are_not_vetoed():
    a = _m("E1", "P01", "Paula Corrales", event="EV:E1")
    b = _m("E2", "P02", "Paula Corrales", event="EV:E9")
    assert not score(a, b, _stats([a, b]))["vetoed"]


def test_assign_event_ids_groups_identical_payloads_only():
    ms = [_m("A", "P01", "X"), _m("B", "P01", "X"), _m("C", "P01", "Y")]
    assert assign_event_ids(ms) == 1
    assert ms[0]["_event"] == ms[1]["_event"]
    assert ms[2].get("_event") is None


def test_assign_event_ids_ignores_entry_text_and_uses_people():
    """Differing transcription over identical people is the case dedupe misses."""
    ms = [_m("A", "P01", "X"), _m("B", "P01", "X")]
    ms[0]["text_faithful"] = "one transcription"
    ms[1]["text_faithful"] = "a slightly different transcription"
    assert assign_event_ids(ms) == 1
    assert ms[0]["_event"] == ms[1]["_event"]


def test_assign_event_ids_is_idempotent():
    ms = [_m("A", "P01", "X"), _m("B", "P01", "X")]
    first = assign_event_ids(ms)
    evs = [m["_event"] for m in ms]
    assert assign_event_ids(ms) == first
    assert [m["_event"] for m in ms] == evs


def test_empty_payload_entries_are_not_grouped():
    """334 entries extract nobody and all hash alike -- grouping them would
    fuse unrelated records into one event, the 380-vs-59 trap again."""
    ms = [_m("A", "P01", "X")]
    assert assign_event_ids(ms) == 0
    assert ms[0].get("_event") is None
