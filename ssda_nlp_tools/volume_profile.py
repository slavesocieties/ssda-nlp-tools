"""What language is this volume, and what sacrament is it a register of?

`build_messages` takes `record_type` and `language` and defaults them to
"baptism" and "Spanish". No caller ever passed them, so every volume extracted
so far was told it was a Spanish baptism register -- including 701054, which is
Portuguese burials, and 29597, which is Spanish marriages.

It did no measured damage. 701054 still produced 187 burials and 29597 723
marriages, because the system prompt forbids translating between Iberian
languages and the model read the actual text rather than the label. But it is a
landmine, and the fix that lasts is not "remember the flags" -- it is to make
the default correct.

MIXED VOLUMES ARE THE INTERESTING CASE and the reason this returns a confidence
rather than a label. 701157 carries 904 baptism signals and 740 marriage ones;
701008 is 777 baptism against 270 burial. Naming either one "baptism" tells the
model most of a volume it is about to read is something it is not. Below a clear
majority the type is reported as "sacramental", which is true of all of them and
misleads about none.

Signals are counted over the raw transcription, not the extraction, so this can
be run before anything has been extracted -- which is the point.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Portuguese-only markers. Deliberately not the shared Iberian vocabulary:
# "sepultura" and "cadaver" are identical in both languages and were what made
# an earlier language detector call five Portuguese entries Spanish.
_PORTUGUESE = re.compile(
    r"\b(?:encommend\w*|sepultou|freguezia|freguesia|obito|óbito|mez\b|"
    r"assent\w*|assign\w*|assinei|vigr\.?o|dezembro|janeiro|fevereiro|"
    r"nasceu|faleceu|idade de|oitocentos|filho legitimo de)\b", re.I)
_SPANISH = re.compile(
    r"\b(?:yglesia|iglesia|parroquial|bautic\w*|bautiz\w*|ochocientos|"
    r"setecientos|dicho\b|hijo legitimo de|vecino|difunto|dias del mes|"
    r"a[nñ]os\b|se[nñ]or|nuestra se[nñ]ora)\b", re.I)

_TYPES = {
    "baptism": re.compile(
        r"\b(?:bapti[sz]\w*|batiz\w*|baut[ií][csz]\w*|santos oleos|"
        r"pila baptismal|padrinho|padrino|madrina|madrinha)\b", re.I),
    "burial": re.compile(
        r"\b(?:sepult\w*|encommend\w*|cad[aá]ver\w*|obito|óbito|enterr\w*|"
        r"falec\w*|falleci\w*|muri[oó]|difunt\w*|cemiterio|cemitério)\b", re.I),
    "marriage": re.compile(
        r"\b(?:matrimoni\w*|amonesta\w*|casad[oa]s? y velad|desposar\w*|"
        r"esposar\w*|banhos|contrajo matrimonio|recebe?r[aã]o em matrim)\b", re.I),
}

# Below this share of type signals, the volume is called "sacramental" rather
# than named for its plurality. 701157 sits at 0.54 and is genuinely mixed.
CLEAR_MAJORITY = 0.75


def profile_text(text: str) -> Dict[str, Any]:
    pt, es = len(_PORTUGUESE.findall(text)), len(_SPANISH.findall(text))
    language = "Portuguese" if pt > es else "Spanish"
    counts = {k: len(rx.findall(text)) for k, rx in _TYPES.items()}
    total = sum(counts.values())
    best = max(counts, key=counts.get) if total else "baptism"
    share = counts[best] / total if total else 0.0
    return {
        "language": language,
        "language_signals": {"Portuguese": pt, "Spanish": es},
        "record_type": best if share >= CLEAR_MAJORITY else "sacramental",
        "dominant_type": best,
        "type_share": round(share, 3),
        "type_signals": counts,
        "mixed": share < CLEAR_MAJORITY,
    }


def profile_entries(entries: List[dict], sample: int = 400) -> Dict[str, Any]:
    """Profile a segmented volume.

    Samples EVENLY ACROSS the volume rather than taking the first N, because
    these books change register partway through. 701008 shows 270 burial signals
    over its full text and ZERO in its first 400 entries: the burials are in a
    later section, and a head sample calls it a pure baptism register. A bound is
    still wanted -- a 2,000-entry volume saturates the signal long before the
    end -- but the bound has to be spread.
    """
    rows = entries or []
    if len(rows) > sample:
        step = len(rows) / sample
        rows = [rows[int(i * step)] for i in range(sample)]
    return profile_text(" ".join((e.get("text") or e.get("raw") or "")
                                 for e in rows))
