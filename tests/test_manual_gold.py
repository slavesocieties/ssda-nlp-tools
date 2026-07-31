"""Tests for measuring machine transcription against hand transcription.

The bugs these hold closed are alignment bugs, not arithmetic ones. Both were
found in real data and both produced plausible-looking accuracy figures rather
than crashing, which is what makes them worth a test each.
"""
import pytest

from ssda_nlp_tools.manual_gold import (align_pages, aggregate, compare_text,
                                        human_page_of, human_pages, machine_page_of,
                                        machine_pages, normalize, offset_map)


# --- page id conventions: the two sources genuinely differ ------------------ #

def test_human_page_is_the_first_group():
    assert human_page_of("0033-01") == "0033"
    assert human_page_of("0006-05") == "0006"


def test_machine_page_is_the_last_group_not_the_volume_id():
    """`15834-0001.jpg` must be page 1, not `1583`.

    One regex served both conventions and a leading \\d{3,4} search matched four
    digits of the VOLUME id, so all of 15834 keyed to a bogus page and aligned to
    nothing. It only failed loudly because the key sets were intersected.
    """
    assert machine_page_of("15834-0001.jpg") == "0001"
    assert machine_page_of("1795-0007.jpg") == "0007"
    assert machine_page_of("419324-0201.jpg") == "0201"
    assert machine_page_of("FHL_007548705-0012.jpg") == "0012"


def test_volume_id_is_never_mistaken_for_a_page():
    for vol in ("1795", "15834", "419324", "701157"):
        assert machine_page_of(f"{vol}-0042.jpg") == "0042"


# --- comparison fairness ---------------------------------------------------- #

def test_line_wrapping_is_not_counted_as_error():
    """Human transcription is diplomatic and splits words across lines."""
    human = "Mil Setecien tos Noventa y Tres exer ci las sacras"
    machine = "Mil Setecientos Noventa y Tres exerci las sacras"
    assert compare_text(human, machine)["cer_nospace"] == 0.0


def test_accents_and_case_are_conventions_not_content():
    assert normalize("Bapticé á María") == normalize("baptice a maria")


def test_identical_text_is_perfect():
    r = compare_text("bautice a Maria hija de Juan", "bautice a Maria hija de Juan")
    assert r["similarity"] == 1.0 and r["cer"] == 0.0


def test_machine_extra_text_counts_as_insertion_not_substitution():
    """The machine reads marginalia and stamps the human skipped. That is scope,
    not error, which is why the rates are reported separately."""
    r = compare_text("bautice a Maria", "N. 335 bautice a Maria [sello]")
    assert r["insert"] > 0 and r["substitute"] == 0


def test_aggregate_is_character_weighted_not_page_averaged():
    """A 60-character margin note must not outvote a 3,000-character folio."""
    big = compare_text("a" * 3000, "a" * 3000)          # perfect, huge
    small = compare_text("bcdefg", "xxxxxx")            # awful, tiny
    agg = aggregate([big, small])
    assert agg["sub_rate"] < 0.01


# --- drift: the bug that made 120 pages look like transcription failures ---- #

"""Distinct per-page content matters here. An earlier version of this fixture
gave every page the same sentence with only the number changed, so misaligned
pages still scored ~1.0 and the drifted baseline looked perfect -- the fixture
hid the bug it was written to demonstrate."""
_NAMES = ["Maria Dolores", "Juan Congo", "Petrona Mandinga", "Lorenzo Noriega",
          "Ysabel Mondongo", "Manuel Pallares", "Thomasa Mina", "Joseph Mellado",
          "Catalina Benigna", "Francisco Sanchez"]
_PLACES = ["Guamacaro", "Matansas", "La Habana", "Cienfuegos", "San Agustin",
           "Camarones", "Cumanayagua", "Santa Clara", "Trinidad", "Regla"]


def _drifting_corpus():
    """Human pages 1-4 line up at +0; pages 5-8 sit at +1, as 15834 really does."""
    texts = {i: (f"en la parroquial de {_PLACES[i % 10]} baptise y puse los santos "
                 f"oleos a {_NAMES[i % 10]} hija natural de {_NAMES[(i + 3) % 10]} "
                 f"vecina de {_PLACES[(i + 5) % 10]} y lo firme")
             for i in range(1, 20)}
    human = {str(i).zfill(4): texts[i] for i in range(1, 9)}
    machine = {}
    for i in range(1, 9):
        tgt = i if i <= 4 else i + 1
        machine[str(tgt).zfill(4)] = texts[i]
    machine["0005"] = "an inserted unnumbered folio of something else entirely"
    return human, machine


def test_offset_map_follows_drift():
    human, machine = _drifting_corpus()
    offs = offset_map(human, machine, window=3, neighbourhood=4)
    assert offs["0002"] == 0
    assert offs["0008"] == 1


def test_align_pages_recovers_drifted_pages():
    human, machine = _drifting_corpus()
    drifted = aggregate(align_pages(human, machine, drift=False)["pages"])
    fixed = aggregate(align_pages(human, machine, drift=True,
                                  window=3, neighbourhood=3)["pages"])
    assert fixed["median_similarity"] > drifted["median_similarity"]
    assert fixed["median_similarity"] > 0.95


def test_a_pages_own_score_never_decides_its_own_offset():
    """Aligning each page to its best match would maximise similarity by
    construction and inflate every accuracy number. The offset comes from
    NEIGHBOURS, so a page that genuinely transcribed badly stays badly scored
    instead of being re-homed to whichever folio it happens to resemble.
    """
    human, machine = _drifting_corpus()
    human["0003"] = "completely unrelated text about a shipment of sugar"
    offs = offset_map(human, machine, window=3, neighbourhood=4)
    assert offs["0003"] == 0          # neighbours say +0, not "wherever it fits"
    rows = {r["page"]: r for r in align_pages(human, machine,
                                             window=3, neighbourhood=3)["pages"]}
    assert rows["0003"]["similarity"] < 0.5


def test_no_drift_when_alignment_is_already_clean():
    texts = {str(i).zfill(4): f"page {i} bautice a Maria hija de Juan"
             for i in range(1, 10)}
    res = align_pages(texts, dict(texts))
    assert res["pages_realigned"] == 0
    assert res["offsets_used"] == {0: 9}


def test_human_pages_rebuilds_page_from_entries_in_order():
    vol = {"entries": [{"id": "0007-02", "raw": "second"},
                       {"id": "0007-01", "raw": "first"},
                       {"id": "0008-01", "raw": "next page"}]}
    pages = human_pages(vol)
    assert pages["0007"] == "first\nsecond"
    assert pages["0008"] == "next page"


def test_machine_pages_reads_the_archivault_shape():
    assert machine_pages([{"file": "1795-0003.jpg", "transcription": "hola"}]) == {
        "0003": "hola"}
