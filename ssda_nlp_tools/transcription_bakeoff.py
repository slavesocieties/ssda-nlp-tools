"""Score a transcription by what the downstream pipeline can do with it.

The problem this solves: comparing two transcriptions of 1660s handwriting has
no ground truth. We hold no human-verified text, so character error rate is not
computable, and comparing model A against model B tells us only that they
differ, not which is right.

But we do not actually care which is *prettier*. We care whether the text
supports the work: whether entries can be found and stitched, whether the
controlled vocabulary survives, whether the router can parse the page. Those are
measurable without ground truth, and they measure the thing that matters. A
transcription with elegant prose and mangled line endings is worse for us than a
rougher one that keeps entry boundaries intact.

Signals, all free and deterministic:

  entries_per_page      the segmenter's yield
  partial_rate          entries left dangling across a page break. This
                        discriminates sharply -- 701179 went 131 partials to 1
                        on a segmentation change alone
  low_confidence_pages  pages the router cannot parse and sends to fallback
  error_pages           embedded transcription-API failures in the text
  formula_rate          share of entries opening with a recognised sacramental
                        formula ("Aos ... dias do mez", "En la Villa de ...").
                        These registers are highly formulaic, so a garbled
                        transcription loses them
  vocab_hits_per_1k     controlled-vocabulary terms appearing verbatim per 1,000
                        words. Mangled words fall out of a closed vocabulary in
                        a way that is countable

None of these is decisive alone; together they are a strong screen. The verdict
on accuracy still needs Daniel reading the divergences.
"""
from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List

# Opening formulae. Deliberately loose -- we are detecting that the shape
# survived transcription, not parsing it.
_FORMULAE = re.compile(
    r"(a?os\s+\w+\s+(dias\s+)?d[eo]\s*(mez|m[êe]s)"      # PT: Aos X dias do mez
    r"|em\s+\w+\s+d[eo]\s+\w+\s+de\s+mil"                 # PT: Em X de Y de mil
    r"|en\s+(la|el)\s+(villa|ciudad|yglesia|iglesia)"     # ES: En la Villa de
    r"|en\s+\w+\s+d[eo]\s+\w+\s+de\s+mil"                 # ES: En X de Y de mil
    r"|aos\s+\w+\s+de\s+\w+\s+de\s+mil"                   # PT: Aos X de Y de mil
    r")", re.I)

# Embedded failure markers the transcription service leaves in the text.
_ERROR_MARKS = re.compile(
    r"(\[?(transcription|api)[ _-]?(error|failed|unavailable)\]?"
    r"|\bERROR\b\s*:"
    r"|<\s*error\s*>"
    r"|unable to (transcribe|process))", re.I)


def _entries(volume: Dict[str, Any]) -> List[dict]:
    if not isinstance(volume, dict):
        raise ValueError("expected a segmented volume JSON object, not raw Archivault page JSON; "
                         "run `run_transcription_bakeoff.py segment` first")
    entries = volume.get("entries") or volume.get("records")
    if not isinstance(entries, list):
        raise ValueError("input has no segmented entries/records; run "
                         "`run_transcription_bakeoff.py segment` first")
    return entries


def _images(entry: dict) -> List[str]:
    """Accept both the production and deterministic segmenter image fields."""
    values = entry.get("source_images") or entry.get("images")
    if values:
        return [str(value) for value in values]
    return [str(entry["image"])] if entry.get("image") else []


def _text(entry: dict) -> str:
    return (entry.get("text") or entry.get("text_faithful")
            or entry.get("raw") or entry.get("normalized") or "")


def vocab_terms(vocab) -> set:
    """Every controlled-vocabulary surface form, folded, long enough to be
    distinctive. Two-character terms would match noise."""
    from .vocab import _fold
    terms = set()
    for field, langs in vocab.by_field.items():
        for values in langs.values():
            for v in values:
                f = _fold(v)
                if len(f) >= 4:
                    terms.add(f)
    return terms


def score_transcription(volume: Dict[str, Any], vocab=None,
                        pages: int = 0) -> Dict[str, Any]:
    """Downstream-usability metrics for one segmented volume."""
    from .vocab import _fold, load_vocab
    vocab = vocab or load_vocab()
    terms = vocab_terms(vocab)

    entries = _entries(volume)
    stats = volume.get("stats") or {}
    n_pages = (pages or stats.get("pages")
               or len({img for e in entries for img in _images(e)})
               or 0)

    texts = [_text(e) for e in entries]
    joined = "\n".join(texts)
    words = re.findall(r"[^\W\d_]+", joined, re.UNICODE)
    folded = {_fold(w) for w in words}

    lengths = sorted(len(t) for t in texts) or [0]
    partial = sum(1 for e in entries if e.get("partial"))
    formula = sum(1 for t in texts if _FORMULAE.search(t))
    errors = len(_ERROR_MARKS.findall(joined))

    return {
        "pages": n_pages,
        "entries": len(entries),
        "entries_per_page": round(len(entries) / n_pages, 3) if n_pages else None,
        "partial": partial,
        "partial_rate": round(partial / len(entries), 4) if entries else None,
        "low_confidence_pages": len(volume.get("low_confidence_pages")
                                    or volume.get("low_confidence") or []),
        "error_pages": len(volume.get("error_pages") or []),
        "error_marks_in_text": errors,
        "chars": len(joined),
        "words": len(words),
        "median_entry_chars": lengths[len(lengths) // 2],
        "formula_rate": round(formula / len(entries), 4) if entries else None,
        "vocab_hits": len(folded & terms),
        "vocab_hits_per_1k_words": round(1000 * len(folded & terms) / len(words), 3)
        if words else None,
    }


# Higher is better for these; lower is better for the rest.
_HIGHER_BETTER = ("entries_per_page", "formula_rate", "vocab_hits_per_1k_words",
                  "median_entry_chars", "words")
_LOWER_BETTER = ("partial_rate", "low_confidence_pages", "error_pages",
                 "error_marks_in_text")


MATERIAL = 0.05          # relative difference below this is a tie, not a win


def compare(a: Dict[str, Any], b: Dict[str, Any],
            label_a: str = "A", label_b: str = "B",
            material: float = MATERIAL) -> Dict[str, Any]:
    """Per-metric winner, plus a tally. Deliberately NOT a single weighted score.

    But a raw win-count is not neutral either, and that error was caught by
    running it: on a pair where one side had 150x the dangling-entry rate, the
    tally came out 3-3, because a 0.09% difference in vocabulary hits was scored
    as a "win" exactly equal to the catastrophe. An unweighted tally is an
    implicit EQUAL weighting, which is a strong claim, not the absence of one.

    So differences below `material` in relative terms are ties. That does not
    rank the metrics against each other -- it only refuses to call a rounding
    difference a victory. Each row keeps its relative delta so the size of a win
    stays visible rather than being flattened into a count.
    """
    rows, wins = [], {label_a: 0, label_b: 0, "tie": 0}
    for metric in _HIGHER_BETTER + _LOWER_BETTER:
        x, y = a.get(metric), b.get(metric)
        if x is None or y is None:
            continue
        scale = max(abs(x), abs(y))
        rel = abs(x - y) / scale if scale else 0.0
        if rel < material:
            winner = "tie"
        elif metric in _HIGHER_BETTER:
            winner = label_a if x > y else label_b
        else:
            winner = label_a if x < y else label_b
        wins[winner] += 1
        rows.append({"metric": metric, label_a: x, label_b: y,
                     "rel_diff": round(rel, 4), "better": winner})
    return {"rows": rows, "wins": wins, "material_threshold": material,
            "verdict": _verdict(wins, label_a, label_b)}


def _verdict(wins, label_a, label_b) -> str:
    decided = wins[label_a] + wins[label_b]
    if not decided:
        return "no measurable difference"
    lead = abs(wins[label_a] - wins[label_b]) / decided
    front = label_a if wins[label_a] > wins[label_b] else label_b
    if lead < 0.25:
        return (f"{front} leads narrowly ({wins[label_a]}-{wins[label_b]}); "
                f"too close to act on without Daniel reading divergences")
    return f"{front} wins {wins[label_a]}-{wins[label_b]} on downstream usability"


def divergent_pages(a: Dict[str, Any], b: Dict[str, Any], top: int = 15):
    """Pages whose two transcriptions differ most, for human adjudication.

    Ranked by dissimilarity so the reviewer's time goes where the models
    actually disagree, not where they already agree.
    """
    from difflib import SequenceMatcher

    def by_page(vol):
        out: Dict[str, List[str]] = {}
        for e in _entries(vol):
            for img in _images(e):
                out.setdefault(str(img), []).append(_text(e))
        return {k: "\n".join(v) for k, v in out.items()}

    pa, pb = by_page(a), by_page(b)
    rows = []
    for img in sorted(set(pa) & set(pb)):
        # autojunk=False: it silently disables matching on texts over 200
        # characters, which is every page here, and reports absurd similarity
        ratio = SequenceMatcher(None, pa[img], pb[img], autojunk=False).ratio()
        rows.append({"image": img, "similarity": round(ratio, 4),
                     "a": pa[img], "b": pb[img]})
    rows.sort(key=lambda r: r["similarity"])
    return rows[:top]


def repair_burden(volume: Dict[str, Any], heavy: float = 0.80) -> Dict[str, Any]:
    """How much the extractor had to REPAIR the transcriber's output.

    A post-extraction metric, so it cannot run in the free `score` pass -- but it
    is the most direct evidence that transcription quality reaches the end of the
    pipeline. The extractor is handed faithful text and returns a normalised
    version; the distance between them is work it had to do because the
    transcription was wrong, and at the extreme it is the extractor refusing:

        "El fragmento contiene partes intercaladas de varios registros y no
         permite una transcripcion ni extraccion fiable."

    That is a whole record lost to a mis-transcribed page, not a style choice.

    Measured on the delivered corpus (gemini-3.1-pro, 5,028 entries): median
    similarity 0.909, and 11.2% heavily rewritten. A better transcriber should
    move both, which makes this a success criterion rather than an impression.
    """
    from difflib import SequenceMatcher
    sims = []
    for e in _entries(volume):
        a = (e.get("text_faithful") or e.get("raw") or "").strip()
        b = (e.get("normalized") or "").strip()
        if len(a) < 80 or len(b) < 80:
            continue                      # too short for a stable ratio
        # autojunk=False: it silently stops matching above 200 characters, which
        # is most entries, and reports absurd similarity
        sims.append(SequenceMatcher(None, a, b, autojunk=False).ratio())
    if not sims:
        return {"entries_compared": 0, "median_similarity": None,
                "heavily_rewritten": None, "heavily_rewritten_rate": None}
    sims.sort()
    n_heavy = sum(1 for s in sims if s < heavy)
    return {
        "entries_compared": len(sims),
        "median_similarity": round(sims[len(sims) // 2], 4),
        "mean_similarity": round(sum(sims) / len(sims), 4),
        "p05_similarity": round(sims[len(sims) // 20], 4),
        "heavily_rewritten": n_heavy,
        "heavily_rewritten_rate": round(n_heavy / len(sims), 4),
    }
