"""Controlled vocabularies (Daniel, 2026-07-24) — canonicalization + conformance.

Before this the extraction prompt deferred every field's semantics to the
few-shot examples, so nothing tied `relationship_type`, `age`, or the
ethnicity/phenotype split to SSDA's actual vocab.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import vocab as V  # noqa: E402
from ssda_nlp_tools import batch_extract as bx  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_relationship_types_map_to_the_closed_english_set():
    v = V.load_vocab()
    english = set(v.values("relationship_type", "English"))
    assert len(english) == 9, "the canonical set is 9 values"
    for surface, want in [("padrino", "godparent"), ("madrina", "godparent"),
                          ("padrinho", "godparent"), ("afilhada", "godchild"),
                          ("esclavo", "slave"), ("escrava", "slave"),
                          ("amo", "enslaver"), ("senhor", "enslaver"),
                          ("hijo", "child"), ("filha", "child"),
                          ("madre", "parent"), ("pai", "parent"),
                          ("esposa", "spouse"), ("nieta", "grandchild")]:
        got = v.canonicalize("relationship_type", surface)
        assert got == want, f"{surface} -> {got!r}, expected {want!r}"
        assert got in english


def test_age_is_a_category_never_a_number_or_a_spanish_word():
    v = V.load_vocab()
    assert set(v.values("age", "English")) == {"infant", "child", "adult"}
    # parvulo/a is the register's word for an infant, and we measured a high
    # hallucination rate on this field before it was specified
    assert v.canonicalize("age", "párvulo") == "infant"
    assert v.canonicalize("age", "parvula") == "infant"
    assert v.canonicalize("age", "criança") == "child"
    assert v.canonicalize("age", "adulta") == "adult"
    assert v.canonicalize("age", "23") is None      # no confident mapping


def test_booleans_read_the_register_s_own_wording():
    v = V.load_vocab()
    assert v.canonicalize("free", "esclava") is False
    assert v.canonicalize("free", "liberto") is True
    assert v.canonicalize("free", "forra") is True
    assert v.canonicalize("legitimate", "legítima") is True
    # "hijo natural" means born out of wedlock
    assert v.canonicalize("legitimate", "natural") is False


def test_source_language_fields_are_never_auto_translated():
    """titles and phenotype are recorded as written; guessing would rewrite the
    record rather than read it."""
    v = V.load_vocab()
    assert v.canonicalize("titles", "Don") is None
    assert v.canonicalize("phenotype", "moreno") is None
    # but they are still validated
    assert v.is_known("titles", "Don")
    assert v.is_known("phenotype", "moreno")


def test_accents_do_not_split_a_vocabulary_value():
    v = V.load_vocab()
    assert v.is_known("ethnicity", "Lucumi") and v.is_known("ethnicity", "Lucumí")
    assert v.canonicalize("rank", "Alferez") == "Ensign"


def test_positional_maps_only_used_where_lists_align():
    """ranks and ethnicity are parallel across languages; the rest are ragged and
    must not be zipped together."""
    v = V.load_vocab()
    assert v.canonicalize("rank", "Teniente Coronel") == "Lieutenant Colonel"
    assert v.canonicalize("ethnicity", "Mandinga") == "Mandinka"
    # occupation lists are ragged (34/35/34) -> no positional guessing
    assert v.canonicalize("occupation", "Escribano") is None


def test_conformance_on_daniels_own_gold_is_near_total():
    """The ceiling. His hand-built examples are what our output is measured
    against, so if they scored poorly the vocabulary would be the wrong yardstick."""
    with open(os.path.join(_ROOT, "training_data.json"), encoding="utf-8") as f:
        td = json.load(f)
    rep = V.check_conformance(td)
    assert rep["relationship_type"]["seen"] > 150
    assert rep["relationship_type"]["rate"] > 0.98
    for field in ("titles", "age", "occupation", "phenotype", "free", "legitimate"):
        assert rep[field]["rate"] == 1.0, f"{field} unexpectedly drifts in gold"


def test_conformance_reports_drift_without_silently_correcting_it():
    records = [{"data": {"people": [
        {"id": "P01", "name": "X", "age": "23", "ethnicity": "criolla",
         "relationships": [{"related_person": "P02", "relationship_type": "owner"}]},
    ], "events": []}}]
    rep = V.check_conformance(records)
    assert rep["age"]["in_vocab"] == 0 and "23" in rep["age"]["stray"]
    assert "owner" in rep["relationship_type"]["stray"]
    # the input is untouched — reporting only
    assert records[0]["data"]["people"][0]["age"] == "23"


def test_extraction_prompt_states_the_closed_sets():
    """The prompt is generated from vocab.json, so it cannot drift from it."""
    p = bx.BATCH_SYSTEM_PROMPT
    for rel in ("spouse", "godparent", "enslaver", "grandchild"):
        assert rel in p
    assert "infant, child, adult" in p
    assert "witnesses" in p                      # was absent from our schema
    assert "ethnicity and phenotype are DIFFERENT" in p
    assert "P01" in p


def test_prompt_stays_byte_identical_across_batches():
    """Cache-ordering depends on the static prefix never varying."""
    a = bx.build_messages([{"entry": "1", "raw": "x"}], [], [])
    b = bx.build_messages([{"entry": "2", "raw": "y"}], [], [])
    assert a[0]["content"] == b[0]["content"] == bx.BATCH_SYSTEM_PROMPT
