"""Triage of the raw QA issue list.

The point of this module is that 1,048 raw issues is not 1,048 defects. These
tests hold the separations that make the number mean something.
"""
import pytest

from audit_corpus import SACRAMENT, century_check


def _entry(eid, text, normalized, events):
    return {"id": eid, "text_faithful": text, "normalized": normalized,
            "data": {"events": events}}


def test_a_cover_page_is_not_a_missing_record():
    """242 of the 299 no_people issues are pages like this. They have no people
    because they are not sacramental records."""
    assert not SACRAMENT.search(
        "OFICIO 1886 y 87 N. 2.299.137 5 C. DE PESO Pbro. Br. D. Clemente Pereira")


def test_a_record_describing_a_sacrament_with_no_people_is_a_defect():
    assert SACRAMENT.search("bautice solemnemente a una nina hija natural de")
    assert SACRAMENT.search("encommendei e sepultou-se o cadaver de Braulia")


def test_century_spelled_in_words_must_match_the_extracted_year():
    bad, checked = century_check(
        [_entry("v-1", "en siete de octubre de mil setecientos setenta y siete",
                "", [{"type": "marriage", "date": "1677-10-07"}])],
        field="text_faithful")
    assert checked == 1 and len(bad) == 1
    assert bad[0]["expected_century"] == "17xx"


def test_a_correct_century_passes():
    bad, checked = century_check(
        [_entry("v-2", "de mil ochocientos ochenta y siete", "",
                [{"type": "baptism", "date": "1887-10-02"}])],
        field="text_faithful")
    assert checked == 1 and bad == []


def test_normalization_drift_is_caught_separately_from_the_date():
    """29597-0257-A-01 reads "mil nov.ta y dos" (1792) and normalizes to "mil
    novecientos noventa y dos" -- 1992, in an 18th-century register. The
    extracted date is CORRECT, so every date-based check passes and only a
    check against the normalized text sees it."""
    e = _entry("v-3", "en tres de Febrero de mil nov.ta y dos a.s",
               "en tres de febrero de mil novecientos noventa y dos",
               [{"type": "marriage", "date": "1792-02-03"}])
    by_text, _ = century_check([e], field="text_faithful")
    by_norm, _ = century_check([e], field="normalized")
    assert by_text == [], "the abbreviated faithful text states no century"
    assert len(by_norm) == 1 and by_norm[0]["expected_century"] == "19xx"


def test_an_entry_with_several_events_passes_if_any_year_matches():
    """A baptism record carrying a later marginal marriage note has two
    centuries in play and is not an error."""
    bad, _ = century_check(
        [_entry("v-4", "de mil ochocientos ochenta y siete", "",
                [{"type": "baptism", "date": "1887-10-02"},
                 {"type": "marriage", "date": "1914-11-26"}])],
        field="text_faithful")
    assert bad == []


def test_ambiguous_or_absent_century_is_skipped_not_guessed():
    for text in ("", "sin fecha legible",
                 "de mil setecientos ... y de mil ochocientos"):
        _, checked = century_check(
            [_entry("v-5", text, "", [{"type": "baptism", "date": "1800-01-01"}])],
            field="text_faithful")
        assert checked == 0, text


def test_events_without_a_parseable_date_are_not_counted():
    _, checked = century_check(
        [_entry("v-6", "de mil ochocientos ochenta", "",
                [{"type": "baptism", "date": None}])],
        field="text_faithful")
    assert checked == 0
