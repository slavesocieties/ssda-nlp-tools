"""`partial` must mean "this record runs off the page" — NOT "we did not
recognise its closing formula".

segment_page is page-local, so it can only infer completeness from the
_CLOSER/_SIGNATURE lexicons, which are harvested per register format. On a
volume whose sign-off wording we have never seen, every page-final record looks
unclosed. That inflated the delivered partial rate to ~11% of the corpus
(Daniel, 2026-07-24: should be <1%). segment_volume now settles it positionally.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import segment  # noqa: E402


def _entry(day, name, closer):
    return (f"En {day} de Enero de mil ochocientos noventa y ocho yo el "
            f"presbitero bauticé solemnemente a {name} parvulo hijo natural "
            f"de padres no conocidos siendo padrinos vecinos de esta ciudad {closer}")


# a sign-off the lexicon has never seen
UNKNOWN_CLOSER = "y asi lo testifico por la presente."


def _vol(pages):
    return segment.segment_volume([(f"900001-{i:04d}.jpg", t)
                                   for i, t in enumerate(pages)])


def test_unknown_closing_formula_does_not_create_a_partial():
    """Page 2 opens its own dated record, so the page-1 record demonstrably
    ended — even though its closing formula is unknown to us."""
    res = _vol([_entry("dos", "Maria Josefa", UNKNOWN_CLOSER),
                _entry("cinco", "Juan Bautista", UNKNOWN_CLOSER),
                _entry("nueve", "Ana Dolores", UNKNOWN_CLOSER)])
    entries = res["entries"]
    assert len(entries) >= 3
    # only the volume-final record may remain unconfirmed
    for e in entries[:-1]:
        assert not e.get("partial"), f"{e['id']} wrongly flagged partial: {e['text'][:80]!r}"


def test_record_ending_on_the_last_page_stays_partial():
    """No next page to confirm against — stay honest and keep the flag."""
    res = _vol([_entry("dos", "Maria Josefa", UNKNOWN_CLOSER),
                _entry("cinco", "Juan Bautista", "")])
    assert res["entries"][-1].get("partial") is True


def test_genuine_run_on_is_still_flagged_and_stitched():
    """A record that really does continue onto the next page keeps partial
    until the continuation closes it, and records both source images."""
    p1 = _entry("dos", "Maria Josefa", "") + " y siendo testigos de este acto"
    p2 = ("los vecinos de esta ciudad que abajo firman "
          + UNKNOWN_CLOSER + "\n"
          + _entry("cinco", "Juan Bautista", UNKNOWN_CLOSER))
    res = _vol([p1, p2, _entry("nueve", "Ana Dolores", UNKNOWN_CLOSER)])
    first = res["entries"][0]
    assert len(first.get("source_images", [])) == 2, "continuation was not stitched"
    assert not first.get("partial"), "stitched record should be resolved complete"


def test_unreadable_next_page_keeps_the_flag():
    """An Archivault failure page cannot confirm anything, so do not clear."""
    res = _vol([_entry("dos", "Maria Josefa", ""),
                "[transcription failed: max retries reached]",
                _entry("nueve", "Ana Dolores", UNKNOWN_CLOSER)])
    assert res["entries"][0].get("partial") is True


def test_partial_rate_is_independent_of_closer_lexicon_coverage():
    """The regression itself: destroying the closing-formula lexicon must not
    change how many records are reported as truncated."""
    pages = [_entry(d, n, UNKNOWN_CLOSER) for d, n in
             (("dos", "Maria Josefa"), ("cinco", "Juan Bautista"),
              ("nueve", "Ana Dolores"), ("once", "Pedro Ignacio"))]
    before = sum(1 for e in _vol(pages)["entries"] if e.get("partial"))

    never = re.compile(r"(?!x)x")
    orig_c, orig_s = segment._CLOSER, segment._SIGNATURE
    segment._CLOSER, segment._SIGNATURE = never, never
    try:
        after = sum(1 for e in _vol(pages)["entries"] if e.get("partial"))
    finally:
        segment._CLOSER, segment._SIGNATURE = orig_c, orig_s

    assert before == after, (
        f"partial count moved {before} -> {after} purely from lexicon coverage")
