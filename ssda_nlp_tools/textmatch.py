"""Name normalization, similarity, and greedy bipartite alignment.

Historical Spanish/Portuguese register names are noisy: accents come and go,
honorifics (Don/Doña) attach and detach, spelling drifts across scribes. We keep
the dependency surface at the stdlib (unicodedata + difflib) so this runs anywhere,
but isolate similarity behind one function so a stronger matcher (rapidfuzz,
jellyfish double-metaphone) can drop in later without touching callers.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Honorifics / titles that should not drive name identity.
_TITLES = {
    "don", "dona", "dna", "d", "sr", "sra", "san", "santa", "sto", "sta",
    "fray", "frai", "padre", "presbitero", "br", "dr", "capitan", "teniente",
}


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_name(name: Optional[str]) -> str:
    """Lowercase, de-accent, drop punctuation and honorifics, collapse spaces."""
    if not name:
        return ""
    s = strip_accents(str(name)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in _TITLES]
    return " ".join(tokens)


def name_tokens(name: Optional[str]) -> List[str]:
    return normalize_name(name).split()


def phonetic_fold(token: str) -> str:
    """Fold one already-normalized token to a Spanish 'sound skeleton'.

    Approximate, not a full double-metaphone, but enough to group scribal
    variants: silent h, v/b and z/s and qu/c->k merges, soft c/g and j -> x,
    ll/y -> i, collapse doubles. Cheap and dependency-free; the natural upgrade
    is jellyfish double-metaphone, swapped in behind this one function.
    """
    t = strip_accents(token).lower()
    out: List[str] = []
    i, n = 0, len(t)
    while i < n:
        c = t[i]
        nxt = t[i + 1] if i + 1 < n else ""
        if c == "h":
            i += 1; continue                     # silent h
        if c == "c":
            if nxt in "ei":
                out.append("s")
            elif nxt == "h":
                out.append("c"); i += 1          # 'ch' = one sound
            else:
                out.append("k")
        elif c == "q":
            out.append("k")
            if nxt == "u":
                i += 1                            # 'qu' -> k, u silent
        elif c == "g":
            out.append("x" if nxt in "ei" else "g")
        elif c in "zx":
            out.append("s")
        elif c in "vw":
            out.append("b")
        elif c == "j":
            out.append("x")
        elif c == "y":
            out.append("i")
        elif c == "l" and nxt == "l":
            out.append("i"); i += 1              # 'll' -> i (yeismo)
        else:
            out.append(c)
        i += 1
    folded = []
    for ch in out:
        if not folded or folded[-1] != ch:       # collapse doubles
            folded.append(ch)
    return "".join(folded)


def phonetic_key(name: Optional[str]) -> str:
    """Blocking key: folded form of the first name token (empty if no name)."""
    toks = name_tokens(name)
    return phonetic_fold(toks[0]) if toks else ""


def name_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Similarity in [0,1] combining token overlap (Jaccard) and edit ratio.

    Token-set Jaccard catches reordered / partial names ("Juan Vives" vs
    "Vives"); SequenceMatcher catches spelling drift ("Matanzas" vs "Matansas").
    We take the max so either signal can rescue a true match, then apply a small
    floor so a shared surname alone is not treated as identity.
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    inter = ta & tb
    jaccard = len(inter) / len(ta | tb) if (ta or tb) else 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()

    # phonetic token overlap catches scribal spelling drift (Vives/Bibes,
    # Gonzalez/Gonzales) that surface forms miss; discounted so sound-alikes
    # alone don't assert identity.
    fa = {phonetic_fold(t) for t in ta}
    fb = {phonetic_fold(t) for t in tb}
    finter = fa & fb
    fjacc = len(finter) / len(fa | fb) if (fa or fb) else 0.0

    # gate: need a shared token (surface or phonetic) OR high edit ratio to count
    if not inter and not finter and ratio < 0.6:
        return 0.0
    return max(jaccard, ratio, 0.92 * fjacc)


def greedy_align(
    gold: Sequence[dict],
    pred: Sequence[dict],
    key: str = "name",
    threshold: float = 0.72,
    score_fn=None,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy max-similarity bipartite matching between two lists of dicts.

    Returns (matches, unmatched_gold_idx, unmatched_pred_idx) where matches is a
    list of (gold_idx, pred_idx, score) with score >= threshold, each item used
    at most once. Greedy (highest score first) is deterministic and good enough
    for the small per-entry cardinalities here; Hungarian is a future swap.
    """
    if score_fn is None:
        score_fn = lambda g, p: name_similarity(g.get(key), p.get(key))

    scored: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            s = score_fn(g, p)
            if s >= threshold:
                scored.append((s, gi, pi))
    scored.sort(reverse=True)

    used_g, used_p = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for s, gi, pi in scored:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, s))

    unmatched_g = [i for i in range(len(gold)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    return matches, unmatched_g, unmatched_p


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision / recall / F1 from a confusion triple."""
    p = tp / (tp + fp) if (tp + fp) else (1.0 if tp == 0 and fp == 0 and fn == 0 else 0.0)
    r = tp / (tp + fn) if (tp + fn) else (1.0 if tp == 0 and fn == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def sum_prf(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Micro-average a set of confusion triples."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return prf(tp, fp, fn)
