"""Local vocabulary extensions (vocab_extensions.json).

Daniel, 2026-07-29, on the 71 unreviewed ethnicity terms: "all terms are valid
and should be added to the vocabulary." These tests pin down what "added" was
implemented to mean, and guard the two ways the merge can fail silently.
"""
import json
import os

import pytest

from ssda_nlp_tools import vocab as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def v():
    return V.load_vocab(reload=True)


def test_the_vendored_file_is_untouched_by_the_extensions():
    """vocab.json is a verbatim copy of slavesocieties/openai. Local additions
    must live in vocab_extensions.json or the next upstream sync silently
    reverts them."""
    raw = json.load(open(os.path.join(ROOT, "vocab.json"), encoding="utf-8"))
    eth = [b for b in raw["controlled_vocabularies"] if b["key"] == "ethnicity"]
    assert {len(b["vocab"]) for b in eth} == {37}       # upstream length, unchanged
    assert not any("asiático" in b["vocab"] for b in eth)


def test_new_terms_keep_the_three_language_lists_index_aligned(v):
    """The positional map is built only when the lists are the same length; if
    an extension appends to English alone the map is dropped and canonicalize()
    returns None for EVERY ethnicity, not just the new one."""
    lengths = {k: len(x) for k, x in v.by_field["ethnicity"].items()}
    assert len(set(lengths.values())) == 1, lengths
    assert v.canonicalize("ethnicity", "Mandinga") == "Mandinka"   # old rows survive
    assert v.canonicalize("ethnicity", "asiático") == "Asian"      # new rows work


def test_a_ragged_extension_raises_instead_of_disabling_the_map():
    ragged = {"ethnicity": {"new_terms": [{"english": "X", "spanish": "X"}]}}
    raw = json.load(open(os.path.join(ROOT, "vocab.json"), encoding="utf-8"))
    with pytest.raises(ValueError, match="missing a language"):
        V.Vocab(raw, ragged)


def test_gender_folding_covers_demonyms_not_just_the_o_a_pair(v):
    """'inglés'/'inglesa' and 'español'/'española' are not -o/-a, so the
    original rule rejected ordinary feminine forms of listed values."""
    assert v.is_known("ethnicity", "inglesa")
    assert v.canonicalize("ethnicity", "inglesa") == "English"
    assert v.canonicalize("ethnicity", "española") == "Spanish"
    assert v.canonicalize("ethnicity", "africana") == "African"


def test_plurals_resolve_to_the_listed_singular(v):
    """-s after a vowel, -es after a consonant. These were three separate
    'unknown terms' in the review queue and are not new ethnicities."""
    assert v.canonicalize("ethnicity", "criollos") == "Creole"
    assert v.canonicalize("ethnicity", "Gangaes") == "Ganga"
    assert v.canonicalize("ethnicity", "Macuaes") == "Makua"


def test_morphology_never_rewrites_the_record(v):
    """Folding is applied to the lookup only. Nothing in vocab coerces a
    scribe's 'africana' to 'africano'."""
    assert v.is_known("ethnicity", "africana")
    assert "africana" not in v.by_field["ethnicity"]["Spanish"]


def test_scribal_variants_resolve_to_the_existing_head_not_a_new_ethnicity(v):
    """Eight renderings of Carabalí are orthography, not eight peoples."""
    for surface in ["Cazabalí", "Casabalí", "Caravali", "Carabaly",
                    "Canabali", "Canabalí", "Cambali"]:
        assert v.canonicalize("ethnicity", surface) == "Carabali", surface
    assert v.by_field["ethnicity"]["English"].count("Carabali") == 1


def test_indio_folds_into_indigenous_but_indio_asiatico_does_not(v):
    """'Tomas Yndio natural de Yucatan' (201991-0279-A-02) is the source-language
    form of a descriptor already listed. 'indio asiático' denotes an Asian
    person and routing it to Indigenous would invert its meaning."""
    assert v.canonicalize("ethnicity", "Indio") == "Indigenous"
    assert v.canonicalize("ethnicity", "india") == "Indigenous"
    assert v.canonicalize("ethnicity", "indio asiático") == "Asian"


def test_every_queued_term_is_now_recognized(v):
    """The whole point of the exercise: nothing in the review queue is still
    counted as off-vocabulary."""
    queue = os.path.join(ROOT, "production", "luna_v3", "ETHNICITY_REVIEW_QUEUE.md")
    terms = [ln[3:].rsplit(" (", 1)[0] for ln in
             open(queue, encoding="utf-8").read().splitlines()
             if ln.startswith("## ")]
    assert len(terms) == 71
    assert [t for t in terms if not v.is_known("ethnicity", t)] == []


def test_terms_added_against_our_own_judgement_stay_visible(v):
    """Four of the 71 look like they belong to another field and three like
    surnames. They were added as instructed, but the flag is what makes the
    decision reversible instead of buried."""
    flagged = {t["english"]: t["flagged"] for t in v.flagged["ethnicity"]}
    assert flagged["Brown"] == "wrong-field"          # a phenotype value
    assert flagged["Augustinian"] == "wrong-field"    # a religious order
    assert flagged["Unknown nation"] == "explicit-null"
    assert flagged["Maroon"] == "status-not-ethnicity"


def test_adding_moreno_costs_us_a_quality_signal(v):
    """Documented consequence, not an accident: 'moreno' is now valid in BOTH
    phenotype and ethnicity, so conformance can no longer detect the extractor
    putting a phenotype term in the ethnicity slot."""
    assert v.is_known("phenotype", "moreno") and v.is_known("ethnicity", "moreno")
