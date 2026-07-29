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
# Local additions live beside the vendored file rather than inside it, so
# vocab.json stays a byte-for-byte copy of upstream and can be re-synced.
DEFAULT_EXTENSIONS_PATH = os.path.join(_HERE, os.pardir, "vocab_extensions.json")

_LANG_KEY = {"english": "English", "spanish": "Spanish", "portuguese": "Portuguese"}

# Spanish/Portuguese gender alternations, as (suffix, counterpart) pairs applied
# in both directions. The -o/-a case was already handled; demonyms mostly are
# not -o/-a ("inglés"/"inglesa", "español"/"española"), so conformance was
# rejecting perfectly ordinary feminine forms of values already in the list.
_GENDER_ALTERNATIONS = (("o", "a"), ("es", "esa"), ("ol", "ola"),
                        ("an", "ana"), ("or", "ora"), ("in", "ina"))


def _gender_variants(folded: str):
    """Every counterpart-gender spelling of a folded surface form."""
    for a, b in _GENDER_ALTERNATIONS:
        if folded.endswith(a):
            yield folded[: -len(a)] + b
        if folded.endswith(b):
            yield folded[: -len(b)] + a


def _singulars(folded: str):
    """Candidate singulars. Spanish/Portuguese pluralise in -s after a vowel and
    -es after a consonant, so both strips are tried: 'criollos' -> 'criollo',
    'Gangaes' -> 'ganga', 'Macuaes' -> 'macua'."""
    if folded.endswith("es") and len(folded) > 3:
        yield folded[:-2]
    if folded.endswith("s") and len(folded) > 2:
        yield folded[:-1]

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

    def __init__(self, raw: Dict[str, Any], extensions: Optional[Dict[str, Any]] = None):
        self.by_field: Dict[str, Dict[str, List[str]]] = {}
        for cv in raw["controlled_vocabularies"]:
            self.by_field.setdefault(cv["key"], {})[cv["language"]] = list(cv["vocab"])

        # local additions, merged BEFORE the positional maps are built below
        self.variants: Dict[str, Dict[str, str]] = {}
        self.flagged: Dict[str, List[Dict[str, Any]]] = {}
        self._apply_extensions(extensions or {})

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

        # Variants are recognized AND resolve to the English head they belong
        # to, so an alias is a real vocabulary entry rather than a value that
        # merely stops being reported.
        for field, mapping in self.variants.items():
            self.allowed.setdefault(field, set()).update(mapping)
            self._to_english.setdefault(field, {}).update(mapping)

    def _apply_extensions(self, ext: Dict[str, Any]) -> None:
        """Merge vocab_extensions.json into the vendored lists.

        New terms are appended at the same index in every language list. For a
        positional field that is the whole contract: append to English only and
        canonicalize() silently returns None for every value in the field,
        because the ragged-list guard above disables the map wholesale.
        """
        for field, spec in ext.items():
            if field.startswith("_") or not isinstance(spec, dict):
                continue
            key = _VOCAB_KEY.get(field, field)
            langs = self.by_field.setdefault(key, {})
            for term in spec.get("new_terms") or []:
                if any(term.get(l) in (None, "") for l in _LANG_KEY):
                    raise ValueError(
                        f"{key}: new term {term!r} is missing a language. Every "
                        f"row must supply all of {sorted(_LANG_KEY)} or the "
                        f"positional alignment breaks.")
                for local, canonical in _LANG_KEY.items():
                    langs.setdefault(canonical, []).append(term[local])
                if term.get("flagged"):
                    self.flagged.setdefault(key, []).append(term)
            if key in _POSITIONAL:
                lengths = {len(v) for v in langs.values()}
                if len(lengths) > 1:
                    raise ValueError(
                        f"{key}: language lists are ragged after extension "
                        f"({ {k: len(v) for k, v in langs.items()} }); "
                        f"canonicalize() would be silently disabled.")
            for head, surfaces in (spec.get("variants") or {}).items():
                for s in surfaces:
                    self.variants.setdefault(key, {})[_fold(s)] = head

    # -- queries ---------------------------------------------------------- #

    def values(self, field: str, language: str = "English") -> List[str]:
        return list(self.by_field.get(_VOCAB_KEY.get(field, field), {}).get(language, []))

    def is_known(self, field: str, value: Any) -> bool:
        key = _VOCAB_KEY.get(field, field)
        allowed = self.allowed.get(key, set())
        folded = _fold(value)
        # Daniel, 2026-07-27 (Q2): fold grammatical gender when CHECKING
        # conformance, rather than coercing records to masculine forms. The
        # record keeps whatever the scribe wrote ("morena", "parda"); only this
        # check is gender-blind. Number is folded the same way (2026-07-29), so
        # "criollos" is the plural of a listed value rather than a new term.
        return self._resolve(folded, allowed) is not None

    @staticmethod
    def _resolve(folded: str, allowed: set) -> Optional[str]:
        """The listed spelling this surface form reduces to, or None.

        Tried in order: as written, counterpart gender, singular, and singular
        in counterpart gender. Morphology is applied to the LOOKUP only; nothing
        rewrites the record.
        """
        for candidate in (folded, *_gender_variants(folded)):
            if candidate in allowed:
                return candidate
        for singular in _singulars(folded):
            for candidate in (singular, *_gender_variants(singular)):
                if candidate in allowed:
                    return candidate
        return None

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
        table = self._to_english.get(key, {})
        got = table.get(_fold(value))
        if got is None:
            # same gender/number folding is_known uses, so "criollas" reaches
            # Creole instead of falling off the map
            resolved = self._resolve(_fold(value), set(table))
            got = table.get(resolved) if resolved else None
        if got is None:
            return None
        if field in BOOLEAN_FIELDS:
            return got == "true"
        return got


_cached: Optional[Vocab] = None


def load_vocab(path: Optional[str] = None, reload: bool = False,
               extensions_path: Optional[str] = None) -> Vocab:
    global _cached
    default = path is None and extensions_path is None
    if _cached is not None and default and not reload:
        return _cached
    p = path or DEFAULT_VOCAB_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ep = extensions_path or DEFAULT_EXTENSIONS_PATH
    ext: Dict[str, Any] = {}
    if os.path.exists(ep):
        with open(ep, "r", encoding="utf-8") as f:
            ext = json.load(f)
    elif extensions_path:                  # explicitly asked for, so must exist
        raise FileNotFoundError(ep)
    v = Vocab(raw, ext)
    if default:
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
