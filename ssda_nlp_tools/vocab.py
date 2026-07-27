"""SSDA controlled vocabularies — loading, canonicalization, conformance.

Source of truth is `vocab.json` from slavesocieties/openai (vendored at the repo
root, as with the other upstream files). Field semantics come from
`training_data_documentation.txt` in the same repo. Daniel pointed us at both on
2026-07-24; before that the extraction prompt deferred all field semantics to
the few-shot examples, so nothing enforced these.

He describes the vocabularies as "representative rather than truly
comprehensive", and that is the right posture for ethnicity/occupation/titles,
which are open-ended by nature. But measured against his OWN gold
(`training_data.json`, 15 examples / 101 people) conformance is 99-100% on every
field — so in practice these behave as closed sets, and drifting off them is a
defect rather than a legitimate extension. `check_conformance` reports drift
instead of rejecting it, so the distinction stays visible.

Per-field handling is NOT uniform (see training_data_documentation.txt):

    titles      source-language, as written in the record
    phenotype   source-language, "usually recorded exactly as they appear"
    ethnicity   the ethnolinguistic descriptor (Angola, Congo, Lucumi...) --
                NOT the physical one; that is phenotype
    rank        translated to its English equivalent
    occupation  generalized (all clergy -> "Cleric") and translated to English
    age         a broad English category: infant (<2y) / child / adult
    legitimate  boolean
    free        boolean
    relationship_type
                English canonical, a closed 9-value set

Only `ranks` and `ethnicity` are index-aligned across the three language lists,
so those are the only ones we map positionally. The small semantically-closed
fields get explicit maps below. The open-ended ones (titles, phenotype,
occupation) are validated but never auto-translated -- guessing there would
invent history.
"""
from __future__ import annotations

import json
import os
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOCAB_PATH = os.path.join(_HERE, os.pardir, "vocab.json")

# vocab.json spells the rank key plurally
_VOCAB_KEY = {"rank": "ranks"}

# fields governed by a controlled vocabulary, in person records
PERSON_FIELDS = ("titles", "rank", "ethnicity", "age", "occupation",
                 "phenotype", "legitimate", "free")

# the language lists are parallel only for these
_POSITIONAL = ("ranks", "ethnicity")

# fields whose value must be emitted as the ENGLISH canonical form
ENGLISH_CANONICAL = ("rank", "age", "occupation", "relationship_type")

# fields that keep the source language's surface form
SOURCE_LANGUAGE = ("titles", "phenotype")


def _fold(s: Any) -> str:
    """Accent-insensitive, case-insensitive key. Scribal accents are noisy and
    'Lucumí'/'Lucumi' must not read as two different ethnicities."""
    if isinstance(s, bool):
        return "true" if s else "false"
    n = unicodedata.normalize("NFKD", str(s).strip().lower())
    return "".join(c for c in n if not unicodedata.combining(c))


# --------------------------------------------------------------------------- #
# explicit source-language -> English canonical maps
# --------------------------------------------------------------------------- #
# Authored from vocab.json's own per-language lists. These fields are small and
# semantically closed, so an explicit map is safer than positional alignment
# (which does not hold for any of them).

_RELATIONSHIP_MAP = {
    "spouse":      ["conyuge", "esposo", "esposa", "marido", "mujer",
                    "conjuge", "mulher"],
    "parent":      ["padre", "madre", "pai", "mae"],
    "child":       ["hijo", "hija", "filho", "filha"],
    "grandparent": ["abuelo", "abuela", "avo"],
    "grandchild":  ["nieto", "nieta", "neto", "neta"],
    "enslaver":    ["esclavizador", "amo", "ama", "escravizador",
                    "senhor", "senhora"],
    "slave":       ["esclavo", "esclava", "escravo", "escrava"],
    "godparent":   ["padrino", "madrina", "padrinho", "madrinha"],
    "godchild":    ["ahijado", "ahijada", "afilhado", "afilhada"],
}

_AGE_MAP = {
    "infant": ["infante", "parvulo", "parvula"],
    "child":  ["nino", "nina", "crianca", "menino", "menina"],
    "adult":  ["adulto", "adulta"],
}

# booleans. NB "natural" (hijo natural) means born out of wedlock -> illegitimate.
_FREE_MAP = {
    "true":  ["libre", "liberto", "liberta", "horro", "horra",
              "livre", "forro", "forra", "free", "freedman", "freedwoman",
              "verdadero", "verdadeiro"],
    "false": ["esclavo", "esclava", "escravo", "escrava", "enslaved",
              "falso"],
}

_LEGITIMATE_MAP = {
    "true":  ["legitimo", "legitima", "legitimate", "verdadero", "verdadeiro"],
    "false": ["ilegitimo", "ilegitima", "illegitimate", "natural", "falso"],
}

_EXPLICIT = {
    "relationship_type": _RELATIONSHIP_MAP,
    "age": _AGE_MAP,
    "free": _FREE_MAP,
    "legitimate": _LEGITIMATE_MAP,
}

BOOLEAN_FIELDS = ("free", "legitimate")


class Vocab:
    """Loaded controlled vocabularies + the canonicalization maps built from them."""

    def __init__(self, raw: Dict[str, Any]):
        self.by_field: Dict[str, Dict[str, List[str]]] = {}
        for cv in raw["controlled_vocabularies"]:
            self.by_field.setdefault(cv["key"], {})[cv["language"]] = list(cv["vocab"])

        # folded lookup sets, per field, across all languages
        self.allowed: Dict[str, set] = {}
        for field, langs in self.by_field.items():
            merged = set()
            for values in langs.values():
                merged |= {_fold(v) for v in values}
            self.allowed[field] = merged

        self._to_english: Dict[str, Dict[str, str]] = {}
        for field, mapping in _EXPLICIT.items():
            m = {}
            for canon, surfaces in mapping.items():
                m[_fold(canon)] = canon
                for s in surfaces:
                    m[_fold(s)] = canon
            self._to_english[field] = m
        for field in _POSITIONAL:                      # ranks, ethnicity
            langs = self.by_field.get(field, {})
            eng = langs.get("English") or []
            m = {}
            for other, values in langs.items():
                if len(values) != len(eng):
                    continue                            # ragged -> refuse to guess
                for src, dst in zip(values, eng):
                    m[_fold(src)] = dst
            self._to_english[field] = m

    # -- queries ---------------------------------------------------------- #

    def values(self, field: str, language: str = "English") -> List[str]:
        return list(self.by_field.get(_VOCAB_KEY.get(field, field), {}).get(language, []))

    def is_known(self, field: str, value: Any) -> bool:
        key = _VOCAB_KEY.get(field, field)
        allowed = self.allowed.get(key, set())
        folded = _fold(value)
        if folded in allowed:
            return True
        # Daniel, 2026-07-27 (Q2): fold grammatical gender when CHECKING
        # conformance, rather than coercing records to masculine forms. The
        # record keeps whatever the scribe wrote ("morena", "parda"); only this
        # check is gender-blind. Spanish/Portuguese adjectival gender is the -o/-a
        # alternation, so try the counterpart form.
        if folded.endswith("a"):
            if folded[:-1] + "o" in allowed:
                return True
        elif folded.endswith("o"):
            if folded[:-1] + "a" in allowed:
                return True
        return False

    def canonicalize(self, field: str, value: Any) -> Optional[str]:
        """Map a source-language surface form to its English canonical value.

        Returns None when we have no confident mapping — callers should keep the
        original rather than invent one. Fields that are *supposed* to stay in
        the source language (titles, phenotype) always return None.
        """
        if value in (None, "", []):
            return None
        if field in SOURCE_LANGUAGE:
            return None
        key = _VOCAB_KEY.get(field, field)
        got = self._to_english.get(key, {}).get(_fold(value))
        if got is None:
            return None
        if field in BOOLEAN_FIELDS:
            return got == "true"
        return got


_cached: Optional[Vocab] = None


def load_vocab(path: Optional[str] = None, reload: bool = False) -> Vocab:
    global _cached
    if _cached is not None and path is None and not reload:
        return _cached
    p = path or DEFAULT_VOCAB_PATH
    with open(p, "r", encoding="utf-8") as f:
        v = Vocab(json.load(f))
    if path is None:
        _cached = v
    return v


# --------------------------------------------------------------------------- #
# conformance reporting
# --------------------------------------------------------------------------- #

def _people(records: Any):
    """Yield person dicts from a volume, an {examples:[...]} training file, or a
    bare list of records — every shape this pipeline passes around."""
    if isinstance(records, dict):
        records = (records.get("examples") or records.get("entries")
                   or records.get("records") or [])
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        data = rec.get("data") or rec
        for p in (data.get("people") or []):
            if isinstance(p, dict):
                yield p


def _events(records: Any):
    if isinstance(records, dict):
        records = (records.get("examples") or records.get("entries")
                   or records.get("records") or [])
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        data = rec.get("data") or rec
        for e in (data.get("events") or []):
            if isinstance(e, dict):
                yield e


def check_conformance(records: Any, vocab: Optional[Vocab] = None) -> Dict[str, Any]:
    """Per-field: how many extracted values sit inside the controlled vocabulary,
    and what the most common strays are.

    Drift is REPORTED, never corrected here — an out-of-vocab ethnicity may be a
    genuine ethnonym the list omits, and silently rewriting it would fabricate
    history. The point is to make the rate visible so the extraction prompt can
    be judged against it.
    """
    v = vocab or load_vocab()
    report: Dict[str, Any] = {}

    for field in PERSON_FIELDS:
        seen = ok = 0
        stray: Counter = Counter()
        for p in _people(records):
            raw = p.get(field)
            if raw in (None, "", []):
                continue
            for item in (raw if isinstance(raw, list) else [raw]):
                if item in (None, ""):
                    continue
                seen += 1
                if v.is_known(field, item):
                    ok += 1
                else:
                    stray[str(item).strip()] += 1
        report[field] = {"seen": seen, "in_vocab": ok,
                         "rate": (ok / seen) if seen else None,
                         "stray": dict(stray.most_common(8))}

    # relationship_type lives inside person.relationships
    seen = ok = canonical = 0
    stray = Counter()
    english = {_fold(x) for x in v.values("relationship_type", "English")}
    for p in _people(records):
        for r in (p.get("relationships") or []):
            if not isinstance(r, dict):
                continue
            t = r.get("relationship_type")
            if t in (None, ""):
                continue
            seen += 1
            if v.is_known("relationship_type", t):
                ok += 1
            else:
                stray[str(t).strip()] += 1
            if _fold(t) in english:
                canonical += 1
    report["relationship_type"] = {
        "seen": seen, "in_vocab": ok, "rate": (ok / seen) if seen else None,
        "english_canonical": canonical,
        "canonical_rate": (canonical / seen) if seen else None,
        "stray": dict(stray.most_common(8)),
    }

    # events: witnesses is specified upstream but was absent from our schema
    ev_total = ev_witnesses = 0
    ev_types: Counter = Counter()
    for e in _events(records):
        ev_total += 1
        ev_types[str(e.get("type", "")).strip().lower()] += 1
        if e.get("witnesses"):
            ev_witnesses += 1
    report["_events"] = {"total": ev_total, "with_witnesses": ev_witnesses,
                         "types": dict(ev_types.most_common())}
    return report


def format_conformance(report: Dict[str, Any]) -> str:
    lines = ["controlled-vocabulary conformance:",
             "  %-18s %6s %7s %7s  %s" % ("field", "seen", "inVocab", "rate", "top strays")]
    for field, s in report.items():
        if field.startswith("_"):
            continue
        rate = "-" if s["rate"] is None else "%.1f%%" % (100.0 * s["rate"])
        strays = ", ".join("%s x%d" % (k, n) for k, n in list(s["stray"].items())[:3])
        lines.append("  %-18s %6d %7d %7s  %s"
                     % (field, s["seen"], s["in_vocab"], rate, strays or "-"))
    rt = report.get("relationship_type") or {}
    if rt.get("canonical_rate") is not None:
        lines.append("  relationship_type using the English canonical set: %.1f%%"
                     % (100.0 * rt["canonical_rate"]))
    ev = report.get("_events") or {}
    if ev.get("total"):
        lines.append("  events: %d total, %d carry witnesses"
                     % (ev["total"], ev["with_witnesses"]))
    return "\n".join(lines)
