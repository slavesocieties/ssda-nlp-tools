"""Machine transcription measured against human transcription.

Daniel, 2026-07-31, pointing at github.com/slavesocieties/openai/tree/main/json:
"That includes manual transcriptions here".

This is the measurement I told him we could not make. Every quality number in
this project so far has been SELF-consistency -- does the text look like a
register, does it segment, does it parse. None of it could answer "is this what
the manuscript says", because answering that needs a human transcription of the
same page, and we had none.

Nine volumes are hand transcribed (3,452 entries). Three of them -- 1795, 15834,
419324 -- are also in the 232-volume Archivault set, so for those three the same
page exists twice: once as Gemini read it, once as a person read it.

WHAT MAKES THIS COMPARISON UNFAIR, AND WHAT IS DONE ABOUT IT
------------------------------------------------------------
The two sides are not trying to do the same thing, and a naive diff punishes the
machine for things nobody would call an error.

  line wrapping   The human transcription is diplomatic: it preserves the
                  manuscript's line breaks, so a word split across two lines is
                  transcribed split -- "Mil Setecien tos Noventa", "exer ci".
                  Gemini emits running text. Comparing raw strings scores every
                  wrapped word as two errors. Whitespace is therefore collapsed,
                  and a second space-insensitive figure isolates how much of the
                  remaining difference is still word-boundary noise.

  segmentation    Human text is per ENTRY, machine text is per PAGE. Entries are
                  concatenated in id order to rebuild the page before comparing.

  page scope      The human transcribed only the entries. Machine transcription
                  includes marginalia, headers and catalogue stamps that are
                  genuinely on the page and genuinely not in the human file.
                  This inflates machine "insertions" and is NOT an error. It is
                  why insertion and deletion rates are reported separately
                  rather than folded into one accuracy number.

So the honest reading of the output is: substitution rate is the quality signal,
deletion rate is the alarm (machine missed text a human read), and insertion
rate is mostly scope difference and needs eyeballing before it means anything.

ALIGNMENT IS VERIFIED, NOT ASSUMED
----------------------------------
Human ids are `PPPP-EE` and machine files are `<vol>-PPPP.jpg`, which LOOK like
the same page numbering. `align_pages` checks that they actually are by scoring
the match, and reports pages whose similarity is so low that the pairing itself
is suspect. A silent off-by-one in page numbering would otherwise produce a
uniformly terrible error rate and look like a transcription failure.
"""
from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Below this, the two texts are not the same page and the pairing is the bug.
SUSPECT_SIMILARITY = 0.30


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def normalize(text: str, *, accents: bool = True, spaces: bool = True) -> str:
    """Fold everything that is a transcription CONVENTION rather than content."""
    t = (text or "").lower()
    if accents:
        t = strip_accents(t)
    t = t.replace("­", "")                    # soft hyphen
    t = re.sub(r"[‐-―]", "-", t)          # dash variants
    t = re.sub(r"[^\w\s-]", " ", t, flags=re.UNICODE)   # punctuation
    if spaces:
        t = re.sub(r"[-\s]+", " ", t)
    return t.strip()


def _opcode_counts(a: str, b: str) -> Dict[str, int]:
    """Character-level edit counts from a to b.

    autojunk=False is mandatory. SequenceMatcher's default treats any character
    appearing in more than 1% of a sequence over 200 items as junk, which for
    prose means the space character and every common vowel are ignored -- it
    silently reports near-identical texts as unrelated. This has bitten this
    project four separate times.
    """
    sm = SequenceMatcher(None, a, b, autojunk=False)
    out = Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out["equal"] += i2 - i1
        elif tag == "replace":
            out["substitute"] += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            out["delete"] += i2 - i1        # in human, missing from machine
        elif tag == "insert":
            out["insert"] += j2 - j1        # in machine, absent from human
    return dict(out)


def compare_text(human: str, machine: str) -> Dict[str, Any]:
    h, m = normalize(human), normalize(machine)
    c = _opcode_counts(h, m)
    n = max(len(h), 1)
    hs, ms = h.replace(" ", ""), m.replace(" ", "")
    cs = _opcode_counts(hs, ms)
    ns = max(len(hs), 1)
    return {
        "human_chars": len(h),
        "machine_chars": len(m),
        "equal": c.get("equal", 0),
        "substitute": c.get("substitute", 0),
        "delete": c.get("delete", 0),
        "insert": c.get("insert", 0),
        # the quality signal
        "sub_rate": round(c.get("substitute", 0) / n, 5),
        # the alarm: text a human read and the machine did not produce
        "del_rate": round(c.get("delete", 0) / n, 5),
        # mostly page-scope difference; read before believing
        "ins_rate": round(c.get("insert", 0) / n, 5),
        "cer": round((c.get("substitute", 0) + c.get("delete", 0)
                      + c.get("insert", 0)) / n, 5),
        # same, ignoring word boundaries entirely: isolates line-wrap noise
        "cer_nospace": round((cs.get("substitute", 0) + cs.get("delete", 0)
                              + cs.get("insert", 0)) / ns, 5),
        "similarity": round(SequenceMatcher(None, h, m, autojunk=False).ratio(), 5),
    }


def human_page_of(entry_id: str) -> Optional[str]:
    """`0033-01` -> `0033`. Page is the FIRST group; there is no volume prefix."""
    m = re.match(r"\s*(\d+)\s*-", str(entry_id or ""))
    return m.group(1).zfill(4) if m else None


def machine_page_of(filename: str) -> Optional[str]:
    """`15834-0001.jpg` -> `0001`. Page is the LAST group, after the volume id.

    The two conventions genuinely differ, and one regex serving both is how this
    broke: a leading `\\d{3,4}` search on `15834-0001.jpg` matches `1583` -- four
    digits of the VOLUME id -- so every page of 15834 keyed to the same bogus
    page and the volume aligned to nothing. Worth noting the failure was loud
    (zero pages compared) only because align_pages intersects the two key sets.
    Had the volume id happened to be four digits, as `1795` is, every page would
    have collapsed onto one key and produced a plausible-looking error rate.
    """
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(filename or ""))
    m = re.search(r"(\d+)\s*$", stem)
    return m.group(1).zfill(4) if m else None


def human_pages(volume: Dict[str, Any]) -> Dict[str, str]:
    """Rebuild page text from entry text, in entry order."""
    by_page: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for e in volume.get("entries") or []:
        pid = human_page_of(e.get("id"))
        if pid:
            by_page[pid].append((str(e.get("id")), e.get("raw") or ""))
    return {p: "\n".join(t for _, t in sorted(v))
            for p, v in by_page.items()}


def machine_pages(pages: Any) -> Dict[str, str]:
    items = pages if isinstance(pages, list) else (pages.get("pages") or [])
    out: Dict[str, str] = {}
    for p in items:
        if not isinstance(p, dict):
            continue
        pid = machine_page_of(p.get("file") or p.get("id") or "")
        if pid:
            out[pid] = p.get("transcription") or p.get("text") or ""
    return out


def _best_offset(human_text: str, machine: Dict[str, str], page: int,
                 window: int) -> Optional[int]:
    cands = []
    for off in range(-window, window + 1):
        t = str(page + off).zfill(4)
        if t in machine:
            cands.append((compare_text(human_text, machine[t])["similarity"], off))
    return max(cands)[1] if cands else None


def offset_map(human: Dict[str, str], machine: Dict[str, str],
               window: int = 6, neighbourhood: int = 10) -> Dict[str, int]:
    """Which machine page each human page really is, allowing for DRIFT.

    Page numbering between the two sources is not a constant offset. Measured on
    15834: human pages 30-69 line up at +0, pages 70-189 at +1, and 190 onward
    back at +0 -- a numbering discontinuity mid-volume, not a transcription
    problem. A single-offset check cannot see this; it reports +0 as globally
    best and silently scores 120 pages against the wrong folio.

    THE BIAS THIS AVOIDS. Aligning each page to its own best-scoring match would
    maximise similarity by construction and inflate every accuracy number:
    whatever the machine produced, we would go find the human page it most
    resembles. So a page's offset is decided by its NEIGHBOURS -- the modal best
    offset over surrounding pages, excluding the page itself -- and then applied.
    Local consensus fixes the alignment; the page's own score never votes on
    where it belongs.
    """
    raw: Dict[int, int] = {}
    for pid, text in human.items():
        off = _best_offset(text, machine, int(pid), window)
        if off is not None:
            raw[int(pid)] = off
    smoothed: Dict[str, int] = {}
    for page in raw:
        votes = [o for p, o in raw.items()
                 if p != page and abs(p - page) <= neighbourhood]
        smoothed[str(page).zfill(4)] = (
            Counter(votes).most_common(1)[0][0] if votes else raw[page])
    return smoothed


def align_pages(human: Dict[str, str], machine: Dict[str, str],
                drift: bool = True, window: int = 6,
                neighbourhood: int = 10) -> Dict[str, Any]:
    """Compare only pages a human actually transcribed, and flag bad pairings.

    `neighbourhood` must be smaller than the run of pages sharing an offset, or
    one region's consensus is imposed on the whole volume.
    """
    offs = offset_map(human, machine, window, neighbourhood) if drift else {}
    rows, suspect, shifted = [], [], 0
    for pid in sorted(human):
        off = offs.get(pid, 0)
        tgt = str(int(pid) + off).zfill(4)
        if tgt not in machine:
            tgt, off = pid, 0
        if tgt not in machine:
            continue
        r = compare_text(human[pid], machine[tgt])
        r["page"] = pid
        r["machine_page"] = tgt
        r["offset"] = off
        shifted += bool(off)
        rows.append(r)
        if r["similarity"] < SUSPECT_SIMILARITY:
            suspect.append(pid)
    return {
        "pages_compared": len(rows),
        "pages_realigned": shifted,
        "offsets_used": dict(Counter(r["offset"] for r in rows)),
        "human_only_pages": sorted(set(human) - set(machine)),
        "machine_only_pages": len(set(machine) - set(human)),
        "suspect_alignment": suspect,
        "pages": rows,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Character-weighted, not page-averaged: a 60-character margin note must
    not carry the same weight as a 3,000-character folio."""
    if not rows:
        return {}
    tot = sum(r["human_chars"] for r in rows) or 1
    agg = {k: sum(r[k] for r in rows)
           for k in ("equal", "substitute", "delete", "insert",
                     "human_chars", "machine_chars")}
    agg["pages"] = len(rows)
    agg["sub_rate"] = round(agg["substitute"] / tot, 5)
    agg["del_rate"] = round(agg["delete"] / tot, 5)
    agg["ins_rate"] = round(agg["insert"] / tot, 5)
    agg["cer"] = round((agg["substitute"] + agg["delete"] + agg["insert"]) / tot, 5)
    # statistics.median, not sorted[len//2]: the latter takes the UPPER middle
    # on an even count, which nudges every reported median up a little.
    agg["median_similarity"] = round(
        statistics.median(r["similarity"] for r in rows), 5)
    agg["median_cer_nospace"] = round(
        statistics.median(r["cer_nospace"] for r in rows), 5)
    return agg


def entry_counts(volume: Dict[str, Any]) -> Dict[str, int]:
    """Human entries per page -- segmentation ground truth, free with the text."""
    c: Counter = Counter()
    for e in volume.get("entries") or []:
        pid = human_page_of(e.get("id"))
        if pid:
            c[pid] += 1
    return dict(c)
