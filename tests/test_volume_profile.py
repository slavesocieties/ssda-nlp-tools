"""Per-volume language and record-type detection.

build_messages defaults to record_type="baptism", language="Spanish" and no
caller ever passed them, so every volume staged so far was labelled a Spanish
baptism register -- 701054 is Portuguese burials and 29597 is Spanish marriages.
Detection exists so the correct value is the default rather than something a
person has to remember.
"""
from ssda_nlp_tools.volume_profile import profile_entries, profile_text

PT_BURIAL = ("Aos desasseis dias do mez de Junho de mil oitocentos e sessenta e "
             "cinco encommendei e sepultou-se o cadaver de Braulia de um anno de "
             "idade no cemiterio desta freguezia e mandei lavrar este assento")
ES_BAPTISM = ("En la Yglesia Parroquial de esta ciudad bautice solemnemente y puse "
              "los santos oleos a una nina hija legitima de vecinos de esta "
              "feligresia, fue su madrina dicha senora, de mil ochocientos")
ES_MARRIAGE = ("En la Ciudad de la Havana habiendose leido las tres canonicas "
               "amonestaciones sin resultar impedimento, contrajo matrimonio y "
               "fueron casados y velados, de mil setecientos")


def test_portuguese_burial_is_not_a_spanish_baptism():
    p = profile_text(PT_BURIAL)
    assert p["language"] == "Portuguese"
    assert p["record_type"] == "burial"


def test_spanish_baptism():
    p = profile_text(ES_BAPTISM)
    assert p["language"] == "Spanish" and p["record_type"] == "baptism"


def test_spanish_marriage_is_not_called_baptism():
    p = profile_text(ES_MARRIAGE)
    assert p["language"] == "Spanish" and p["record_type"] == "marriage"


def test_shared_iberian_vocabulary_does_not_decide_the_language():
    """`sepultura` and `cadaver` are identical in both languages. An earlier
    detector matched on exactly those and called five Portuguese entries
    Spanish, so neither appears in the language patterns."""
    from ssda_nlp_tools.volume_profile import _PORTUGUESE, _SPANISH
    for shared in ("sepultura", "cadaver"):
        assert not _PORTUGUESE.search(shared), shared
        assert not _SPANISH.search(shared), shared


def test_a_mixed_volume_is_called_sacramental_not_its_plurality():
    """701157 is 904 baptism signals against 740 marriage. Naming it "baptism"
    tells the model most of what it is about to read is something it is not."""
    p = profile_text(ES_BAPTISM * 5 + ES_MARRIAGE * 5)
    assert p["mixed"] and p["record_type"] == "sacramental"
    assert p["dominant_type"] in ("baptism", "marriage")


def test_a_clear_majority_keeps_its_name():
    p = profile_text(ES_BAPTISM * 20 + ES_MARRIAGE)
    assert not p["mixed"] and p["record_type"] == "baptism"


def test_the_sample_is_spread_not_taken_from_the_head():
    """These books change register partway. 701008 shows 270 burial signals over
    its full text and ZERO in its first 400 entries, so a head sample calls it a
    pure baptism register."""
    entries = ([{"text": ES_BAPTISM}] * 800) + ([{"text": PT_BURIAL}] * 800)
    p = profile_entries(entries, sample=400)
    assert p["type_signals"]["burial"] > 0, "spread sample missed the tail section"


def test_empty_volume_does_not_crash():
    p = profile_entries([])
    assert p["record_type"] in ("baptism", "sacramental")
