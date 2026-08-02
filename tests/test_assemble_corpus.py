"""Volume mapping and withdrawal persistence in assemble_corpus.

Both bugs here share a shape: they do not raise, they subtract. An unmapped
custom_id is not an error, it is an absence, and a resurrected record is not a
crash, it is a fabrication back in the delivered data. Neither shows up in a
diff you are not already looking at.
"""
import json

import pytest

from assemble_corpus import _volume_of, apply_delivery_convention


# --- volume mapping: this has silently discarded paid work twice ------------ #

@pytest.mark.parametrize("cid,expected", [
    # batch requests
    ("v3-176899-b0000", "176899"),
    ("v3-29597-b0012", "29597"),
    # repair requests are addressed to an ENTRY, so the volume is followed by a
    # page number rather than by -b0. 160 of these mapped to nothing, and
    # re-assembling dropped the corpus from 5,226 records to 5,066.
    ("v3-repair1-176899-0236-B-01", "176899"),
    ("v3-repair1-29597-0012-A-03", "29597"),
    ("v3-repair1-375062-0100-05", "375062"),
    ("v3-repair1-701054-0004-B-02", "701054"),
    # explicit repair suffix
    ("v3-701054-repair-0001", "701054"),
])
def test_every_real_custom_id_shape_maps_to_its_volume(cid, expected):
    assert _volume_of(cid) == expected


def test_a_page_number_is_never_mistaken_for_a_volume():
    """`0236` is four digits and sits next to the volume. If the pattern is
    loosened further, this is what breaks first."""
    assert _volume_of("v3-repair1-176899-0236-B-01") != "0236"


def test_run_prefix_is_not_a_volume():
    assert _volume_of("v3-176899-b0000") == "176899"


def test_vocabtest_maps_to_none_on_purpose():
    """The prompt experiment reuses delivered entry ids; assembling it would
    collide with the real records."""
    assert _volume_of("v3-701054-vocabtest-b0001") is None


def test_unmappable_id_returns_none_rather_than_guessing():
    assert _volume_of("garbage-without-a-volume") is None


# --- withdrawal must survive a rebuild -------------------------------------- #

def test_withdrawn_ids_are_removed_from_entries():
    """withdraw_records.py edits the materialized file, but assembly rebuilds it
    from source. Without re-applying, the next assembly resurrects two records
    whose text is the model apologising rather than the manuscript."""
    entries = [{"id": "201991-0001-A-01"}, {"id": "201991-0304-A-05"},
               {"id": "201991-0002-A-01"}]
    withdrawn = {"201991-0304-A-05"}
    kept = [e for e in entries if str(e.get("id")) not in withdrawn]
    assert [e["id"] for e in kept] == ["201991-0001-A-01", "201991-0002-A-01"]


def test_delivery_convention_drops_partials_reversibly():
    entries = [{"id": "a", "partial": True}, {"id": "b"}]
    kept, dropped = apply_delivery_convention(entries, keep_partials=False)
    assert [e["id"] for e in kept] == ["b"] and dropped == 1
    kept, dropped = apply_delivery_convention(entries, keep_partials=True)
    assert len(kept) == 2 and dropped == 0
