"""Two failure classes that survive every automated check we have.

Both were found by inspecting the composition of 201991, not by a test, and
both produce well-formed output -- which is exactly why nothing caught them.
"""
import importlib.util

import pytest


def _mod():
    spec = importlib.util.spec_from_file_location("fbr", "find_bad_records.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_a_model_apology_in_the_faithful_text_is_caught():
    """This reached DELIVERY: 201991-0304-A-05 reads "En la Yglesia Parroquial
    de Ntra. Senora de la Asuncion I cannot fulfill this request. I am
    programmed to be a helpful and harmless AI assistant." That is fabricated
    text in a historical record."""
    R = _mod().REFUSAL
    for s in ["I cannot fulfill this request",
              "I am programmed to be a helpful and harmless AI assistant",
              "I'm sorry, but I cannot transcribe the text in this image",
              "As an AI language model I am unable to read this",
              "Lo siento, no puedo transcribir"]:
        assert R.search(s), s


def test_the_bakeoff_pattern_would_have_missed_these():
    """Why the existing check did not fire: it looks for 'transcription error'
    and 'unable to process', not for a first-person apology."""
    from ssda_nlp_tools.transcription_bakeoff import _ERROR_MARKS
    apology = "I cannot fulfill this request. I am programmed to be helpful."
    assert not _ERROR_MARKS.search(apology)
    assert _mod().REFUSAL.search(apology)


def test_ordinary_spanish_is_not_flagged_as_a_refusal():
    """A false positive here sends a historian to re-read a page for nothing."""
    R = _mod().REFUSAL
    for s in ["En la Villa de Guanabacoa se dio sepultura al cadaver de Jose",
              "no pudo recibir los Santos Sacramentos por lo violento de su muerte",
              "no consta el nombre de sus padres"]:
        assert not R.search(s), s


def test_a_short_event_less_record_is_a_margin_note_not_a_miss():
    """353 of the event-less records are marginalia like "f. 53. Ma. de la
    Concepn." Flagging those would bury the 11 real misses in noise."""
    mod = _mod()
    assert len("fa Criolla escl. de Casas") < mod.MIN_CHARS
    assert mod.SACRAMENT.search("se dio sepultura al cadaver")


def test_scan_separates_the_two_classes_because_the_fixes_differ(tmp_path):
    """A refusal needs RE-TRANSCRIPTION -- the source text is wrong. A missed
    event needs RE-EXTRACTION only -- the source text is fine."""
    import json
    mod = _mod()
    vol = {"entries": [
        {"id": "V-1", "text_faithful": "x" * 500 + " se dio sepultura al cadaver de Jose",
         "data": {"people": [], "events": []}},
        {"id": "V-2", "text_faithful": "En la Yglesia I cannot fulfill this request.",
         "data": {"people": [], "events": [{"type": "burial"}]}},
        {"id": "V-3", "text_faithful": "f. 53. Ma. de la Concepn.",
         "data": {"people": [], "events": []}},
    ]}
    p = tmp_path / "201991.materialized.json"
    p.write_text(json.dumps(vol), encoding="utf-8")
    refusal, no_event, short_none, total = mod.scan([str(p)])
    assert [r["id"] for r in refusal] == ["V-2"]
    assert [r["id"] for r in no_event] == ["V-1"]
    assert short_none == 1 and total == 3
