"""A gate that fires when the transcriber gave up, and only then.

Two pages reached delivery whose text was the model's apology rather than the
manuscript. This runs before segmentation on every page, always.

The precision numbers below are measured against all 62,320 transcribed pages,
not asserted, because the first version of this pattern was WRONG on real data.
"""
import pytest

from ssda_nlp_tools.transcription_integrity import (LEGITIMATE_GAPS, check_page,
                                                    check_volume, format_report)

DELIVERED = [
    "En la Yglesia Parroquial de Ntra. Senora de la Asuncion I cannot fulfill "
    "this request. I am programmed to be a helpful and harmless AI assistant.",
    "se enterro el cadaver a Jose de la Maria I'm sorry, but I cannot transcribe "
    "the text in this image. The handwriting is extremely faded.",
]
CORPUS_HITS = [
    "The image provided is too blurry and the handwriting is too faded and "
    "illegible to produce a reliable transcription.",
    "Therefore, I cannot provide a transcription for this image.",
    "As an AI language model, I am unable to read this document.",
]
# Real register text. Every one of these was a FALSE POSITIVE in the first
# version, found by running it over the corpus rather than by review.
REAL_TEXT = [
    "baptizavi infantem, filium legitimum Antonii Cantar parrochiae Sti. Philippi",
    "no puedo conformarme con ella por que no esta arreglada a lo que V. S. mando",
    "En la Villa de Guanabacoa se dio sepultura al cadaver de Jose de nacion Congo",
    "Aos treze dias do mes de Maio... O Vigr.o Manoel Luis dos Reis Caval.o",
    "Maria Cantarero, hija de Antonio Cantar y de Josefa Costa",
    "lo siento no consta el nombre de sus padres",
]


@pytest.mark.parametrize("text", DELIVERED + CORPUS_HITS)
def test_a_transcriber_giving_up_is_caught(text):
    assert not check_page(text)["ok"]
    assert check_page(text)["codes"] == ["refusal"]


@pytest.mark.parametrize("text", REAL_TEXT)
def test_real_register_text_is_never_flagged(text):
    """`no puedo` and `lo siento` are ordinary period Spanish; "Antonii Cantar"
    is a surname. The corpus is Spanish, Portuguese and Latin, so the signal is
    ENGLISH first-person modal language appearing where no English belongs."""
    assert check_page(text)["ok"], text


def test_scribal_illegibility_markers_are_not_failures():
    """A faithful transcription of a damaged folio is FULL of these. Flagging
    them would invert the tool: it would reject the honest transcriptions."""
    for gap in LEGITIMATE_GAPS:
        assert check_page(f"En la Villa de {gap} se dio sepultura al cadaver")["ok"]


def test_a_surname_cannot_trip_the_contraction_pattern():
    """`I can't` requires a following verb precisely so that "Antonii Cantar"
    and "Cantarero" cannot match."""
    assert check_page("filium legitimum Antonii Cantar de Corsiga")["ok"]
    assert not check_page("I can't transcribe this page")["ok"]


def test_volume_report_counts_and_never_edits():
    """A page we cannot read is a fact to report, not a record to quietly drop."""
    pages = [{"file": "a.jpg", "transcription": "En la Villa de Guanabacoa..."},
             {"file": "b.jpg", "transcription": DELIVERED[0]}]
    rep = check_volume(pages)
    assert rep["pages"] == 2 and rep["failed"] == 1
    assert rep["by_code"]["refusal"] == 1
    assert rep["failures"][0]["page"] == "b.jpg"
    assert pages[1]["transcription"] == DELIVERED[0]      # untouched
    assert "not the manuscript" in format_report(rep)


def test_a_clean_volume_reports_clean():
    rep = check_volume([{"file": "a.jpg", "transcription": "Aos treze dias do mes"}])
    assert rep["failed"] == 0 and "no problems" in format_report(rep)
