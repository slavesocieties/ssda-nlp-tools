"""The relationship queue must never let a capped page read as complete."""
import json
import os

from run_relationship_review import DIMENSIONS, collect, render


def _fixture():
    entries = {"E1": {"id": "E1", "text_faithful": "En la iglesia ...",
                      "data": {"people": [{"id": "P01", "name": "Ana",
                                           "relationships": [{"related_person": "P09",
                                                              "relationship_type": "parent"}]}]}}}
    buckets = {"dangling_relationship": [{"entry": "E1", "detail": "missing P09"}] * 5,
               "null_relationship": [{"entry": "E1", "detail": "missing None"}] * 2}
    return entries, buckets


def test_the_cap_is_applied_per_kind():
    entries, buckets = _fixture()
    assert len(collect(entries, buckets, 2)) == 4        # 2 of each kind
    assert len(collect(entries, buckets, 500)) == 7      # everything


def test_rows_carry_the_transcription_so_the_page_is_decidable():
    entries, buckets = _fixture()
    row = collect(entries, buckets, 1)[0]
    assert row["text"].startswith("En la iglesia")
    assert row["people"][0]["relationships"]


def test_the_page_is_self_contained_and_has_no_stray_characters():
    """A typo'd hex colour once left an Arabic letter inside the CSS."""
    entries, buckets = _fixture()
    page = render(collect(entries, buckets, 5), {"dangling_relationship": 5}, 5)
    assert "http://" not in page and "https://" not in page
    assert all(ord(c) < 0x2100 for c in page), "non-latin character in the page source"


def test_html_in_the_transcription_is_escaped():
    entries = {"E1": {"id": "E1", "text_faithful": "<script>alert(1)</script>",
                      "data": {"people": []}}}
    buckets = {"dangling_relationship": [{"entry": "E1", "detail": "<b>x</b>"}]}
    page = render(collect(entries, buckets, 1), {}, 1)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
