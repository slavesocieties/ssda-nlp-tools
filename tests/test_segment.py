"""Offline tests for entry segmentation (Task 3) against the REAL gold pairs."""
import json
import os
import re
import sys

from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools.segment import (
    classify_line, join_lines, load_pages, segment_page, segment_volume)
from ssda_nlp_tools.segeval import evaluate_segmentation, margin_number_check

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
ROOT = os.path.dirname(HERE)


def _nsp(s):
    return re.sub(r"\s+", "", s).lower()


def _sim(a, b):
    return SequenceMatcher(None, _nsp(a), _nsp(b)).ratio()


# ---- the two authoritative gold pairs ----------------------------------------

def _run_pair(n):
    ext = "md" if n == 1 else "json"
    gold = json.load(open(os.path.join(FIX, f"output_{n}.json"), encoding="utf-8"))
    pages = load_pages(os.path.join(FIX, f"input_{n}.{ext}"))
    pred = segment_page(pages[0][1], image=gold["image"])
    return gold, pred


def test_gold_pair_1_portuguese_baptisms_with_trailing_partial():
    gold, pred = _run_pair(1)
    assert len(pred["entries"]) == len(gold["entries"]) == 4
    for g, p in zip(gold["entries"], pred["entries"]):
        assert _sim(g["text"], p["text"]) >= 0.92
        assert g["partial"] == p["partial"]
    assert pred["entries"][-1]["partial"] is True          # the trailing fragment


def test_gold_pair_2_portuguese_burials_with_interleaved_margins():
    gold, pred = _run_pair(2)
    assert len(pred["entries"]) == len(gold["entries"]) == 4
    for g, p in zip(gold["entries"], pred["entries"]):
        assert _sim(g["text"], p["text"]) >= 0.92
        assert g["partial"] == p["partial"]


def test_margin_prefix_stripped_from_opener():
    _, pred = _run_pair(1)
    assert pred["entries"][2]["text"].startswith("Aos vinte e cinco")   # not "Pedro Aos"


# ---- Daniel's manual examples (newer gold format: flat list + images[]) --------

def _run_manual(stem):
    """New-format gold: a flat list of {id, text, images}. Score BOUNDARIES, since
    the gold also silently corrects the transcription (see below) and therefore
    contains text absent from the input — no segmenter can match it exactly."""
    from ssda_nlp_tools.segment import segment_volume
    gold = json.load(open(os.path.join(FIX, f"{stem}_sample_output.json"), encoding="utf-8"))
    pages = load_pages(os.path.join(FIX, f"{stem}_sample.json"))
    pred = segment_volume(pages)["entries"]
    return gold, pred


def _proper_names(s):
    """Capitalized tokens, accent-stripped. These survive the gold's
    normalization; abbreviations and archaic spellings do not."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return {t.lower() for t in re.findall(r"\b[A-Z][a-z]{3,}", s)}


def _boundary_hits(gold, pred, min_shared=2):
    """Did we locate each gold record?

    Matched on shared proper names, NOT on text similarity — deliberately. The
    manual gold is segmentation + normalization: it expands abbreviations
    ("R.do P.e" -> "Reverendo Padre"), modernizes archaic spelling
    ("dezasete" -> "dezessete"), strips the interleaved margin column, and even
    repairs characters Archivault dropped ("ochocien" -> "ochocientos"). A
    segmenter cannot and should not reproduce any of that, so character
    similarity measures the normalizer, not the splitter. Names are the part of
    the record that survives both sides.
    """
    used, hits = set(), 0
    for g in gold:
        gn = _proper_names(g["text"])
        best, bi = 0, None
        for i, p in enumerate(pred):
            if i in used:
                continue
            n = len(gn & _proper_names(p["text"]))
            if n > best:
                best, bi = n, i
        if bi is not None and best >= min_shared:
            used.add(bi); hits += 1
    return hits


def test_manual_gold_65858_portuguese_all_boundaries_found():
    gold, pred = _run_manual("65858")
    assert len(gold) == 10
    assert _boundary_hits(gold, pred) == 10      # every record located


def test_manual_gold_420550_colombia_all_boundaries_found():
    gold, pred = _run_manual("420550")
    assert len(gold) == 8
    assert len(pred) == 8
    assert _boundary_hits(gold, pred) == 8       # "En la parroquia de … á tres de Abril" opener


def test_manual_gold_contains_corrections_absent_from_the_raw():
    """Guards the finding that the gold is not a pure segmentation target: it
    repairs Gemini's dropped characters, so exact-text scoring is not meaningful
    and boundary scoring is the right metric."""
    pages = load_pages(os.path.join(FIX, "420550_sample.json"))
    raw = _nsp(" ".join(t for _, t in pages))
    gold = json.load(open(os.path.join(FIX, "420550_sample_output.json"), encoding="utf-8"))
    gold_txt = _nsp(" ".join(g["text"] for g in gold))
    # Gemini wrote "ochocien noventa"; the gold writes "ochocientos noventa"
    assert "ochocienn" in raw                      # raw is missing "tos"
    assert "ochocientosnoventa" in gold_txt        # gold silently inserts it
    assert "ochocienn" not in gold_txt


def test_manual_gold_is_normalized_not_raw_segmentation():
    """The manual examples are the target for segmentation + NORMALIZATION, not
    segmentation alone: the gold expands abbreviations and modernizes archaic
    spelling, which a rule-based splitter cannot and should not attempt. Pinning
    this so nobody 'fixes' the segmenter to chase an unreachable text score."""
    pages = load_pages(os.path.join(FIX, "740018_sample.json"))
    raw = " ".join(t for _, t in pages)
    gold = json.load(open(os.path.join(FIX, "740018_sample_output.json"), encoding="utf-8"))
    gtxt = " ".join(g["text"] for g in gold)
    # abbreviations present in the transcription, expanded in the gold
    for abbrev, expanded in (("R.do P.e", "Reverendo Padre"), ("leg.mo", "legítimo"),
                             ("fuer.n", "fueron"), ("Padr.os", "Padrinos")):
        assert abbrev in raw, abbrev
        assert abbrev not in gtxt
        assert expanded in gtxt, expanded


def test_740018_all_eleven_records_found_including_demil_year():
    """18th-c. Spanish, heavy abbreviation, margin column, and a first record with
    NO date opener. Also covers the 'demil' regression: the transcriber runs
    'de mil' together, which used to leave the opener 'weak' so it never split
    and a whole record was silently swallowed."""
    from ssda_nlp_tools.segment import segment_volume
    gold = json.load(open(os.path.join(FIX, "740018_sample_output.json"), encoding="utf-8"))
    pages = load_pages(os.path.join(FIX, "740018_sample.json"))
    pred = segment_volume(pages)["entries"]
    assert len(gold) == 11
    assert len(pred) == 11                       # was 10 before the demil fix
    # 1:1 in order, verified via proper names (they survive normalization; the
    # abbreviations do not, so token/text similarity is useless here)
    def names(s):
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return {t.lower() for t in re.findall(r"\b[A-Z][a-z]{3,}", s)}
    for g, p in zip(gold, pred):
        assert len(names(g["text"]) & names(p["text"])) >= 2


def test_partial_means_runs_off_the_page_not_missing_opener():
    """A leading fragment with no date opener is NOT automatically partial. Some
    registers open a record without a date (740018-0006-01); the manual gold
    counts those complete. 'partial' must mean 'the text runs off the page', so
    a fragment carrying its own closer/signature is complete — while a genuinely
    truncated trailing record stays partial."""
    from ssda_nlp_tools.segment import segment_volume
    p740 = segment_volume(load_pages(os.path.join(FIX, "740018_sample.json")))["entries"]
    assert p740[0]["partial"] is False        # has its closing formula + signature
    p658 = segment_volume(load_pages(os.path.join(FIX, "65858_sample.json")))["entries"]
    assert p658[-1]["partial"] is True        # really does continue onto page 0005


def test_demil_counts_as_a_year_marker():
    from ssda_nlp_tools.segment import _YEARISH
    assert _YEARISH.search("En dies y seis de Maio demil sett.os y veinte iocho")
    assert _YEARISH.search("de mil setecientos")
    assert not _YEARISH.search("en la ciudad de Santa Cruz")


# ---- line classification rules ------------------------------------------------

def test_junk_lines():
    for line in ("1793.", "pag. 40", "47..", "3.", "Julho nada", "Agosto",
                 "=ma", "=ceno", "+ Miercoles"):
        assert classify_line(line, "between") == "junk", line


def test_margin_name_junk_only_between_entries():
    assert classify_line("Maria", "between") == "junk"
    assert classify_line("Maria", "inside") == "text"      # interleaved -> keep


def test_opener_detection_incl_split_month():
    assert classify_line("Em vinte edous de Junho de mil Sete Centos",
                         "between") == "opener_strong"
    # month broken at the line wrap: needs the lookahead
    assert classify_line("Domingo, dia veinte y quatro de Noviem", "between",
                         lookahead="bre de Mil, Setecientos, Noventa y tres. Yo Don"
                         ) == "opener_strong"


def test_margin_line_above_opener_does_not_become_the_opener():
    # "Friay." + next line "Domingo, dia..." — the margin line must stay junk
    assert classify_line("Friay.", "between",
                         lookahead="Domingo, dia quinze de Junio de Mil,") == "junk"


def test_el_dia_opener_only_counts_with_a_year():
    # Cienfuegos-style opener ("El dia … de mil ochocientos…") IS an entry start
    assert classify_line("El dia treinta y uno de Julio de mil ochocientos ochenta",
                         "between") == "opener_strong"
    assert classify_line("N.o 3. El dia treinta y uno de Julio de mil ochocientos",
                         "between") == "opener_strong"
    # but an in-body birth date ("nació el dia … del presente año", no year) must
    # NOT be treated as a new entry — this was the false-split we had to guard
    assert classify_line("nació el dia quinze de Junio del presente año, hija",
                         "inside") == "text"
    assert classify_line("que nació el dia quatro del presente mes, y Año,",
                         "inside") == "text"


def test_el_dia_recovers_a_whole_page_of_entries():
    # a Cienfuegos-format page: three "El dia … de mil …" baptisms in a row
    body = ("N. 1.\nJose Felipe.\nEl dia treinta y uno de Julio de mil ochocientos "
            "ochenta y siete, bautice a un nino Jose Felipe, hijo de Lutgardo. "
            "Y para que conste lo firmamos = Francisco Angulo\n"
            "N.o 2.\nMaria.\nEl dia primero de Agosto de mil ochocientos ochenta y "
            "siete, bautice a una nina Maria. Y para que conste lo firmamos = "
            "Francisco Angulo")
    pred = segment_page(body, image="176899-0003.jpg")
    assert len(pred["entries"]) == 2


def test_mid_entry_date_mention_does_not_split():
    text = ("Aos vinte e quatro de Dezembro de mil oitocentos e cincoenta e oito, "
            "Baptisei solemnimente, e pus os Santos Oleos a Maria,\n"
            "nascida a vinte e dous de Novembro do dito anno, filha legitima\n"
            "de Manoel Joao; forao Padrinhos Manoel Joao. E para constar fiz este "
            "assento, que assigno.\nO Vigr.o Joaquim Conrado")
    pred = segment_page(text, image="x.jpg")
    assert len(pred["entries"]) == 1                       # birth date != new entry


def test_join_lines_continuation_marks():
    assert join_lines(["Setecien", "-tos, Noventa"]) == "Setecientos, Noventa"
    assert join_lines(["y la puse por nom", "=bre Dionicia"]) == "y la puse por nombre Dionicia"
    assert join_lines(["Yo Don Mig=", "=uel O Reilly"]) == "Yo Don Miguel O Reilly"
    # end-"=" as a STOP before the signature must NOT glue
    assert join_lines(["y lo firme en dicho dia, mes, y Año=", "Thomas Hassett"]) \
        == "y lo firme en dicho dia, mes, y Año= Thomas Hassett"


# ---- cross-page stitching -------------------------------------------------------

def test_volume_mode_stitches_cross_page_entry():
    p1 = ("Em vinte e dous de Junho de mil Sete Centos Setenta e tres faleceo "
          "da Vida prezente Bernardo menor filho de Bernardo da Silva foy por mim")
    p2 = ("encomendado e Sepultado no Cemiterio desta Matriz de q fiz este asento\n"
          "O Vigr.o Joze da Costa Lopes\n"
          "Em tres de Julho de mil Sete Centos Setenta e tres faleceo da Vida "
          "prezente Faustina preta forra foy por mim encomendada e Sepultada de q "
          "fiz este asento\nO Vigr.o Joze da Costa Lopes")
    res = segment_volume([("v-0001.jpg", p1), ("v-0002.jpg", p2)])
    assert res["stats"]["entries"] == 2
    first = res["entries"][0]
    assert first["partial"] is False                       # stitched to completion
    assert first["source_images"] == ["v-0001.jpg", "v-0002.jpg"]
    assert "encomendado e Sepultado" in first["text"]


def test_catchword_dropped_not_emitted():
    pred = segment_page("=ma\nEm vinte de Junho de mil Sete Centos Setenta "
                        "faleceo Bernardo de q fiz este asento\nO Vigr.o Joze",
                        image="x.jpg")
    assert len(pred["entries"]) == 1
    assert pred["leading_fragment"] is None


# ---- real-volume regression guards ---------------------------------------------

def test_spanish_volume_chunk1_perfect_vs_reference():
    from ssda_nlp_tools.segeval import load_reference_entries
    pages = load_pages(os.path.join(ROOT, "Text data/SSDA_0013_0023_Gemini_V2.json"))
    res = segment_volume(pages)
    ref = load_reference_entries(
        os.path.join(ROOT, "Sample_output/Generated_0013_0023_4o_prompt_V2.json"))
    rep = evaluate_segmentation(ref, res["entries"])
    assert rep["recall"] >= 0.95
    assert rep["coverage_recall"] >= 0.99


def test_structural_margin_agreement_100pct():
    total = agree = 0
    for tag in ("0013_0023", "0024_0034", "0035_0044"):
        pages = load_pages(os.path.join(ROOT, f"Text data/SSDA_{tag}_Gemini_V2.json"))
        res = segment_volume(pages)
        mc = margin_number_check(pages, res["per_image"])
        total += mc["pages"]; agree += mc["agree"]
    assert agree == total == 32
