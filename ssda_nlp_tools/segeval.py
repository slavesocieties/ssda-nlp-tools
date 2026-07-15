"""Evaluate a segmenter's output against reference entries.

Alignment is SPACE-INSENSITIVE (all whitespace stripped before comparison):
the gold pairs themselves are inconsistent about joining line-broken words
("oito centos" vs "oitocentos"), and that style variance must not read as a
content error. Greedy 1:1 matching, then per-volume recall / precision plus
split/merge diagnostics (an unmatched prediction that covers TWO references is
a missed split; two predictions inside one reference is an over-split).
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def _nsp(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _sim(a: str, b: str) -> float:
    na, nb = _nsp(a), _nsp(b)
    if not na or not nb:
        return 0.0
    # cheap length gate before the quadratic matcher
    if abs(len(na) - len(nb)) > 0.6 * max(len(na), len(nb)):
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _contains(hay: str, needle: str, threshold: float = 0.85,
              min_needle: int = 30) -> bool:
    """Is `needle` (approximately) a substring of `hay`? Windowed ratio.
    Needles shorter than `min_needle` chars match everything and mean nothing."""
    H, N = _nsp(hay), _nsp(needle)
    if not H or len(N) < min_needle or len(N) > len(H) + 40:
        return False
    m = SequenceMatcher(None, H, N).find_longest_match(0, len(H), 0, len(N))
    return m.size >= threshold * len(N)


def _page_of_ref(r: dict) -> Optional[str]:
    m = re.match(r"^(\d{3,4})-", str(r.get("id", "")))
    return m.group(1) if m else None


def _pages_of_pred(p: dict) -> set:
    out = set()
    for img in p.get("source_images", []) or []:
        m = re.search(r"-(\d{3,4})\.", img)
        if m:
            out.add(m.group(1))
    m = re.search(r"-(\d{3,4})-\d+$", str(p.get("id", "")))
    if m:
        out.add(m.group(1))
    return out


def evaluate_segmentation(reference: List[dict], predicted: List[dict],
                          threshold: float = 0.75) -> Dict[str, Any]:
    """reference/predicted: [{"text": ...}, ...] (ids optional).

    Matching is PAGE-CONSTRAINED first: registers are full of near-identical
    formulaic entries (same priest, same day), so a same-page pass must run
    before the global pass or duplicate references steal cross-page matches.
    """
    scored: List[Tuple[float, int, int, bool]] = []
    for ri, r in enumerate(reference):
        rp = _page_of_ref(r)
        for pi, p in enumerate(predicted):
            s = _sim(r.get("text") or r.get("raw") or "", p["text"])
            if s >= threshold:
                same_page = bool(rp) and rp in _pages_of_pred(p)
                scored.append((s, ri, pi, same_page))
    used_r, used_p, matches = set(), set(), []
    for pass_same_page in (True, False):
        for s, ri, pi, sp in sorted(scored, reverse=True):
            if sp != pass_same_page or ri in used_r or pi in used_p:
                continue
            used_r.add(ri); used_p.add(pi)
            matches.append({"ref": ri, "pred": pi, "sim": round(s, 3)})

    unmatched_r = [i for i in range(len(reference)) if i not in used_r]
    unmatched_p = [i for i in range(len(predicted)) if i not in used_p]

    # boundary partial-credit: a prediction flagged partial (entry continues on a
    # page/chunk we don't have) whose text IS the visible part of a reference
    # entry is correct segmentation, not an error.
    boundary_partials = []
    for pi in list(unmatched_p):
        p = predicted[pi]
        if not p.get("partial"):
            continue
        for ri in list(unmatched_r):
            rtext = reference[ri].get("text") or reference[ri].get("raw") or ""
            if _contains(rtext, p["text"]) or _contains(p["text"], rtext):
                boundary_partials.append({"ref": ri, "pred": pi})
                used_r.add(ri); used_p.add(pi)
                unmatched_r.remove(ri); unmatched_p.remove(pi)
                break

    # diagnostics on the leftovers
    missed_splits, over_splits, junk_preds, lost_refs = [], [], [], []
    for pi in unmatched_p:
        covered = [i for i in unmatched_r
                   if _contains(predicted[pi]["text"],
                                reference[i].get("text") or reference[i].get("raw") or "")]
        if len(covered) >= 2:
            missed_splits.append({"pred": pi, "covers_refs": covered})
        elif not covered:
            junk_preds.append(pi)
    claimed = {i for ms in missed_splits for i in ms["covers_refs"]}
    for ri in unmatched_r:
        if ri in claimed:
            continue
        parts = [pi for pi in unmatched_p
                 if _contains(reference[ri].get("text") or reference[ri].get("raw") or "",
                              predicted[pi]["text"])]
        if len(parts) >= 2:
            over_splits.append({"ref": ri, "split_into_preds": parts})
        else:
            lost_refs.append(ri)

    n_r, n_p = len(reference), len(predicted)
    n_ok = len(matches) + len(boundary_partials)
    recall = n_ok / n_r if n_r else 1.0
    precision = n_ok / n_p if n_p else 1.0
    # coverage: reference content present SOMEWHERE in the predictions (matched
    # 1:1, boundary partial, or contained in a stitched/merged prediction)
    covered = n_ok
    for ri in unmatched_r:
        rtext = reference[ri].get("text") or reference[ri].get("raw") or ""
        if any(_contains(p["text"], rtext, threshold=0.8) for p in predicted):
            covered += 1
    return {
        "reference": n_r, "predicted": n_p, "matched": len(matches),
        "boundary_partials": boundary_partials,
        "coverage_recall": round(covered / n_r, 4) if n_r else 1.0,
        "recall": round(recall, 4), "precision": round(precision, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
              if precision + recall else 0.0,
        "mean_sim": round(sum(m["sim"] for m in matches) / len(matches), 4) if matches else 0.0,
        "missed_splits": missed_splits, "over_splits": over_splits,
        "junk_predictions": junk_preds, "lost_references": lost_refs,
        "matches": matches,
    }


def load_reference_entries(path: str, drop_duplicates: bool = True,
                           dedupe_sim: float = 0.85) -> List[dict]:
    """Reference from a generated volume ({"examples": [...]}). Optionally drop
    duplicates: QA-confirmed window-overlap dupes AND raw text near-dupes (the
    LLM-repair reference re-emits the same record with small margin variations,
    which would otherwise count as unreachable extra 'entries')."""
    d = json.load(open(path, encoding="utf-8"))
    rows = d.get("examples") or d.get("entries") or []
    entries = [{"id": str(r.get("entry") or r.get("id") or i),
                "text": r.get("raw") or r.get("normalized") or ""}
               for i, r in enumerate(rows)]
    if not drop_duplicates:
        return entries
    from .qa import qa_volume
    rep = qa_volume(d)
    dup_b = {dup["b"] for dup in rep["duplicates"] if dup["confidence"] == "confirmed"}
    entries = [e for e in entries if e["id"] not in dup_b]
    kept: List[dict] = []
    for e in entries:
        if any(_sim(e["text"], k["text"]) >= dedupe_sim for k in kept):
            continue
        kept.append(e)
    return kept


def margin_number_check(pages: List[Tuple[str, str]], per_image: List[dict]) -> Dict[str, Any]:
    """Structural validation against the register's own MARGIN NUMBERS.

    Volume 239746 numbers every entry in the margin ("33..", "41.", "58. ...").
    Those numbers are objective ground truth for how many entries START on each
    page — independent of any LLM reference. Compares that count to the
    segmenter's per-page entry count (leading continuations excluded).
    """
    num_standalone = re.compile(r"^\s*(\d{1,3})\.{0,2}\s*$")
    num_inline = re.compile(r"^\s*(\d{1,3})\.{1,2}\s+\S")
    rows, agree = [], 0
    for (img, text), pg in zip(pages, per_image):
        nums = []
        for ln in text.splitlines():
            m = num_standalone.match(ln) or num_inline.match(ln)
            if m and 1 <= int(m.group(1)) <= 300:
                nums.append(int(m.group(1)))
        expected = len(nums)
        got = len(pg["entries"])
        ok = expected == got
        agree += ok
        rows.append({"image": img, "margin_numbers": nums,
                     "expected_starts": expected, "predicted_starts": got, "ok": ok})
    return {"pages": len(rows), "agree": agree,
            "agreement": round(agree / len(rows), 4) if rows else 1.0,
            "rows": rows}


def format_segeval(rep: Dict[str, Any], tag: str = "") -> str:
    L = [f"--- segmentation eval {tag} ---",
         f"reference entries: {rep['reference']}  predicted: {rep['predicted']}  "
         f"matched: {rep['matched']}"
         + (f" (+{len(rep['boundary_partials'])} boundary partials)"
            if rep.get("boundary_partials") else ""),
         f"recall {rep['recall']:.3f}  precision {rep['precision']:.3f}  "
         f"F1 {rep['f1']:.3f}  coverage {rep['coverage_recall']:.3f}  "
         f"(mean sim {rep['mean_sim']:.3f})"]
    if rep["missed_splits"]:
        L.append(f"missed splits (one pred covers several refs): {len(rep['missed_splits'])}")
    if rep["over_splits"]:
        L.append(f"over-splits (one ref split into several preds): {len(rep['over_splits'])}")
    if rep["junk_predictions"]:
        L.append(f"junk predictions: {len(rep['junk_predictions'])}")
    if rep["lost_references"]:
        L.append(f"lost references: {len(rep['lost_references'])}")
    return "\n".join(L)
