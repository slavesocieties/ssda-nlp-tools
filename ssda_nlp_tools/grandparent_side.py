"""Which side of the family a grandparent belongs to.

Daniel Genkins, 2026-08-10, on the merge scorer treating one record that names
the maternal pair and another that names the paternal pair as four mutually
contradictory grandparents:

    "This is a question that can/should be resolved upstream. Maternal/paternal
     grandparents should be labeled differently when extracted."

So the controlled vocabulary now carries `maternal grandparent` and `paternal
grandparent` beside the plain term (see vocab_extensions.json), and
`evidence.MAX_HOLDERS` drops from a blunt 4 to 2 per side -- which is what makes
a grandparent clash as informative as a parent clash instead of nearly
unreachable.

THE SIDE WAS ALREADY IN THE TEXT; ONLY THE SCHEMA THREW IT AWAY. The registers
state it in a small set of fixed formulae --

    "Abuelos paternos, los pardos Miguel Bazan y Belen de la Rosa"
    "abuelos paternos Don Sebastian Chatelain y Teresa Perez, y maternos Don
     Juan Calvo y Serafina Perez"
    "nieto paterno de Jose y de Belen Sequeira y materno de Pio y de Rosario
     Marias"
    "Abuela materna Manuela criolla"

-- and 1,400 of the 1,431 delivered entries that carry a grandparent edge
(97.8%) contain an explicit materno/paterno cue. Portuguese uses the same two
adjectives, so one parser covers both languages.

That makes this module do double duty:

  * `label_examples` relabels the few-shot demonstrations at prompt-assembly
    time. Without it the prompt would ask for a sided label while every worked
    example showed an unsided one, and the examples would win. Daniel's
    `training_data.json` is vendored verbatim from slavesocieties/openai and
    stays that way -- the same rule vocab.json follows -- so the relabel happens
    in memory rather than in the file.
  * `label_entry` backfills ALREADY-EXTRACTED records from their own source
    text at no API cost, which is the alternative to re-extracting 6,794
    delivered entries to gain a field the transcription already contains.

It is deliberately conservative. A grandparent whose name does not fall inside
exactly one side clause keeps the unsided `grandparent` label; guessing a side
would fabricate a lineage. The unsided term stays in the vocabulary precisely so
there is somewhere honest to put those, and `evidence` still applies the old
capacity-4 rule across the three terms together so an unsided record compared
against a sided one is no worse off than before.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .textmatch import name_tokens

# The two sided terms, and the unsided fallback they refine.
UNSIDED = "grandparent"
MATERNAL = "maternal grandparent"
PATERNAL = "paternal grandparent"
GRANDPARENT_TERMS = (UNSIDED, MATERNAL, PATERNAL)

# Spanish and Portuguese share these adjectives, in both genders and numbers.
_SIDE_WORDS = {
    "paterno": PATERNAL, "paterna": PATERNAL,
    "paternos": PATERNAL, "paternas": PATERNAL,
    "materno": MATERNAL, "materna": MATERNAL,
    "maternos": MATERNAL, "maternas": MATERNAL,
}
_SIDE_RE = re.compile(r"\b(" + "|".join(sorted(_SIDE_WORDS, key=len, reverse=True))
                      + r")\b")

# A side clause runs until the next side word or the end of its own sentence.
# Nothing subtler is needed: the formula is short and always ends at a full stop
# or before the godparents ("Fueron sus padrinos ...") begin.
#
# A COLON IS NOT A STOP. "Abuelos maternos: Justo Montero y Petrona Consuegra"
# is the single commonest form in 176899, and treating ":" as a boundary ended
# every one of those clauses one character after the cue word, leaving nothing
# to match names against.
_STOP_RE = re.compile(r"[.;]|\bfueron sus (?:padrinos|padrinhos)\b"
                      r"|\bforam (?:seus )?padrinhos\b")

# Tokens that carry no identity on their own. "Belen de la Rosa" must match a
# clause on `belen` and `rosa`, not on `de` and `la`.
_STOPWORDS = {"de", "del", "la", "las", "lo", "los", "y", "e",
              "da", "do", "das", "dos"}


def _fold(text: str) -> str:
    """Lowercase and de-accent WITHOUT changing the string's length.

    Offsets found in the folded text have to point at the same characters in the
    original, so the usual NFKD-and-drop-combining-marks fold is unusable here.
    """
    out = []
    for ch in text:
        if ch.isalpha():
            decomposed = unicodedata.normalize("NFKD", ch)
            out.append(decomposed[0].lower())
        else:
            out.append(ch)
    return "".join(out)


def _content_tokens(name: Optional[str]) -> List[str]:
    return [t for t in name_tokens(name) if t not in _STOPWORDS]


def side_clauses(text: str) -> List[Tuple[str, str]]:
    """Split a transcription into [(side, clause-text), ...].

    One clause per materno/paterno cue, running to the next cue or to the end of
    the sentence, whichever comes first. Clauses never overlap, so a name found
    in two of them is genuinely ambiguous rather than double-counted.
    """
    if not text:
        return []
    folded = _fold(text)
    starts = [(m.start(), _SIDE_WORDS[m.group(1)]) for m in _SIDE_RE.finditer(folded)]
    clauses: List[Tuple[str, str]] = []
    for i, (pos, side) in enumerate(starts):
        limit = starts[i + 1][0] if i + 1 < len(starts) else len(folded)
        stop = _STOP_RE.search(folded, min(pos + 1, limit), limit)
        end = stop.start() if stop else limit
        clauses.append((side, folded[pos:end]))
    return clauses


# Why a grandparent could not be sided. Daniel needs these told apart: the first
# two are the register's silence and are nobody's fault, the third is a naming
# collision the text cannot resolve, and the fourth is almost always an
# EXTRACTION DEFECT -- an edge pointing at someone the side clauses never name,
# usually the godparent. Lumping them into one "unresolved" count would hide a
# quality signal inside a coverage number.
NO_SIDE_CLAUSE = "no_side_clause"      # the text never says maternal or paternal
NO_NAME = "no_name"                    # the related person has no usable name
AMBIGUOUS = "name_in_both_clauses"     # e.g. both grandmothers are Maria Antonia
NOT_NAMED = "name_in_no_clause"        # side clauses exist; this person is in none
REASONS = (NO_SIDE_CLAUSE, NO_NAME, AMBIGUOUS, NOT_NAMED)


def classify(name: Optional[str],
             clauses: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """(side, reason-it-failed). Exactly one of the two is None.

    Two passes, both requiring a UNIQUE hit:
      1. every content token of the name appears in exactly one clause. This is
         the normal case and is strict enough that a shared surname across the
         two sides cannot decide it on its own.
      2. failing that, the given name alone appears in exactly one clause. This
         recovers "Abuela materna Manuela criolla", where extraction supplies a
         surname the clause does not.
    """
    if not clauses:
        return None, NO_SIDE_CLAUSE
    toks = _content_tokens(name)
    if not toks:
        return None, NO_NAME

    for required in (toks, toks[:1]):
        hits = {side for side, clause in clauses
                if all(re.search(r"\b" + re.escape(t) + r"\b", clause)
                       for t in required)}
        if len(hits) == 1:
            return hits.pop(), None
        if hits:
            return None, AMBIGUOUS      # named on both sides: refuse to choose
    return None, NOT_NAMED


def side_of(name: Optional[str], clauses: List[Tuple[str, str]]) -> Optional[str]:
    """The side whose clause names this person, or None if that is not certain."""
    return classify(name, clauses)[0]


def _blank_stats() -> Dict[str, Any]:
    s: Dict[str, Any] = {"seen": 0, "maternal": 0, "paternal": 0, "unresolved": 0}
    s.update({r: 0 for r in REASONS})
    s["unresolved_detail"] = []          # [(grandparent-name, reason), ...]
    return s


def label_data(data: Dict[str, Any], text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Relabel a single record's grandparent edges from its own source text.

    Only the edge POINTING AT the grandparent is sided. The reciprocal stays a
    plain "grandchild": the side describes which of the child's parents the line
    runs through, so it belongs to the descendant's view of the relationship and
    the grandparent has no side of their own to state.

    The returned stats break the failures down by `REASONS`, and carry the
    (name, reason) pairs so an audit can name the specific records rather than
    quoting a coverage percentage.
    """
    stats = _blank_stats()
    if not isinstance(data, dict):
        return data, stats

    people = data.get("people") or []
    if not any(str(r.get("relationship_type", "")).lower() in GRANDPARENT_TERMS
               for p in people if isinstance(p, dict)
               for r in (p.get("relationships") or []) if isinstance(r, dict)):
        return data, stats

    data = copy.deepcopy(data)
    people = data.get("people") or []
    clauses = side_clauses(text)
    name_of = {str(p.get("id")): p.get("name") for p in people if isinstance(p, dict)}

    for p in people:
        if not isinstance(p, dict):
            continue
        for r in p.get("relationships") or []:
            if not isinstance(r, dict):
                continue
            if str(r.get("relationship_type", "")).lower() != UNSIDED:
                continue
            stats["seen"] += 1
            name = name_of.get(str(r.get("related_person")))
            side, reason = classify(name, clauses)
            if side is None:
                stats["unresolved"] += 1
                stats[reason] += 1
                stats["unresolved_detail"].append((name, reason))
                continue
            r["relationship_type"] = side
            stats["maternal" if side == MATERNAL else "paternal"] += 1
    return data, stats


def _entry_text(entry: Dict[str, Any]) -> str:
    """The best transcription available on an entry, whatever shape it is in.

    Normalized text first -- accents and expanded abbreviations make the cue
    easier to find -- then the faithful segmenter output, which is what an entry
    has before the extraction pass has run.
    """
    for key in ("text_normalized", "normalized", "text_faithful", "text", "raw"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def label_entry(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`label_data` for a whole entry row, reading the text off the row itself."""
    data = entry.get("data")
    if not isinstance(data, dict):
        return entry, _blank_stats()
    labelled, stats = label_data(data, _entry_text(entry))
    if labelled is data:
        return entry, stats
    entry = dict(entry)
    entry["data"] = labelled
    return entry, stats


def label_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Relabel the few-shot pool so the demonstrations match what we ask for.

    Six of Daniel's fifteen gold examples carry grandparent edges, and every one
    of their source texts states the side ("nieto paterno de ... y materno de
    ..."). Left alone they would teach the model to discard exactly the
    distinction the prompt requests.
    """
    return [label_entry(ex)[0] if isinstance(ex, dict) else ex for ex in examples]
