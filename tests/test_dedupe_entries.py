"""De-duplication: the rails that stop it destroying the corpus."""
import json

from dedupe_entries import dedupe_people_within, n_people, payload_hash


def _entry(eid, people, events=None):
    return {"entry": eid, "data": {"people": people, "events": events or []}}


def _p(pid, name, rels=None):
    return {"id": pid, "name": name, "relationships": rels or []}


# --- the rail that matters most -------------------------------------------

def test_empty_payloads_all_hash_alike():
    """This is WHY emptiness must be excluded, not an incidental detail.

    334 of 6,794 real entries have no extracted people. They are failed
    extractions, not copies of one record, and they are byte-identical to each
    other. Grouping by hash alone would collapse them and destroy ~327 real
    entries, so the caller must gate on n_people() first.
    """
    a, b = _entry("v-1", []), _entry("v-2", [])
    assert payload_hash(a) == payload_hash(b)
    assert n_people(a) == 0 and n_people(b) == 0


def test_distinct_records_do_not_collide():
    a = _entry("v-1", [_p("P01", "Maria")])
    b = _entry("v-2", [_p("P01", "Josefa")])
    assert payload_hash(a) != payload_hash(b)


# --- within-entry duplicate people ----------------------------------------

def test_a_repeated_id_is_collapsed():
    """A person id is the entry's primary key; two people sharing one is an
    extraction error by definition."""
    e = _entry("701157-0214-01", [_p("P01", "Fernando Jose da Costa",
                                     [{"relationship_type": "spouse",
                                       "related_person": "P02"}]),
                                  _p("P02", "Leopoldina"),
                                  _p("P01", "Fernando Jose da Costa")])
    keep, dupes, conflicts = dedupe_people_within(e)
    assert dupes == ["P01"] and conflicts == []
    assert len(keep) == 2
    assert len({p["id"] for p in keep}) == 2


def test_collapsing_never_loses_a_relationship():
    """Duplicates are usually PARTIAL -- the second copy of Fernando carries no
    spouse edge. Whichever copy is kept, the union must survive; an exact-match
    rule missed this case entirely."""
    rel = {"relationship_type": "spouse", "related_person": "P02"}
    # the edge on the SECOND copy, i.e. the one a naive "keep the first" drops
    e = _entry("x", [_p("P01", "Fernando"), _p("P01", "Fernando", [rel])])
    keep, _, _ = dedupe_people_within(e)
    assert len(keep) == 1
    assert rel in keep[0]["relationships"]


def test_relationships_are_not_duplicated_when_both_copies_have_them():
    rel = {"relationship_type": "spouse", "related_person": "P02"}
    e = _entry("x", [_p("P01", "Fernando", [rel]), _p("P01", "Fernando", [rel])])
    keep, _, _ = dedupe_people_within(e)
    assert keep[0]["relationships"] == [rel]


def test_same_id_different_names_is_reported_not_merged():
    """Picking one would silently discard a person. That is a different error
    and needs a human, so both are kept."""
    e = _entry("x", [_p("P01", "Maria"), _p("P01", "Josefa")])
    keep, _, conflicts = dedupe_people_within(e)
    assert len(keep) == 2, "a name conflict must not collapse two people into one"
    assert conflicts and conflicts[0]["names"] == ["Maria", "Josefa"]


def test_an_unnamed_duplicate_inherits_the_name():
    e = _entry("x", [_p("P01", None), _p("P01", "Fernando")])
    keep, _, conflicts = dedupe_people_within(e)
    assert conflicts == []
    assert keep[0]["name"] == "Fernando"


def test_a_clean_entry_is_untouched():
    people = [_p("P01", "Maria"), _p("P02", "Josefa")]
    e = _entry("x", people)
    keep, dupes, conflicts = dedupe_people_within(e)
    assert dupes == [] and conflicts == []
    assert [p["id"] for p in keep] == ["P01", "P02"]
    assert json.dumps(keep, sort_keys=True) == json.dumps(people, sort_keys=True)


# --- redirects: de-duplication must not orphan a graded label --------------

def test_redirects_follow_a_dropped_entry_to_its_survivor(tmp_path):
    """Promoting the de-duplicated corpus orphaned 8 of the 300 pairs Daniel had
    graded, because their entry id no longer existed. A resolver that finds
    nothing returns fewer rows rather than complaining, so this was silent."""
    import json as _json
    from dedupe_entries import load_redirects, resolve_entry
    rep = tmp_path / "r.json"
    rep.write_text(_json.dumps({"collapsed_entries": [
        {"kept": "V-1", "dropped": "V-2"},
        {"kept": "V-1", "dropped": "V-3"},
    ]}), encoding="utf-8")
    red = load_redirects(str(rep))
    assert resolve_entry("V-2", red) == "V-1"
    assert resolve_entry("V-3", red) == "V-1"
    assert resolve_entry("V-1", red) == "V-1", "a surviving id maps to itself"
    assert resolve_entry("V-9", red) == "V-9", "an unknown id is left alone"


def test_redirects_terminate_on_a_cycle():
    """A malformed report must not hang the resolver."""
    from dedupe_entries import resolve_entry
    assert resolve_entry("A", {"A": "B", "B": "A"}) in {"A", "B"}


def test_missing_report_is_not_fatal():
    """Tools must still run on a corpus that was never de-duplicated."""
    from dedupe_entries import load_redirects
    assert load_redirects("does/not/exist.json") == {}
