"""Is this page a complete transcription, or did the transcriber give up?

Runs BEFORE segmentation, on every page, always. A page that fails here is not
a hard page to parse -- it is a page whose text is not the manuscript, and
everything downstream inherits that silently. Two such pages reached delivery
before this existed, one reading:

    "En la Yglesia Parroquial de Ntra. Senora de la Asuncion I cannot fulfill
     this request. I am programmed to be a helpful and harmless AI assistant."

WHY THE PATTERN IS ENGLISH-ONLY, WHICH LOOKS WRONG AND IS NOT
-------------------------------------------------------------
The obvious first version matched Spanish and Portuguese apologies too. Measured
against all 62,320 transcribed pages, it flagged 11 -- and six were WRONG:

    "baptizavi infantem, filium legitimum Antonii Cantar"   <- "i Cant"
    "no puedo conformarme con ella"                         <- ordinary 1840s
                                                               administrative
                                                               Spanish

`no puedo` and `lo siento` are period language in these registers. The corpus is
Spanish, Portuguese and Latin, so what actually distinguishes a model refusal is
ENGLISH first-person modal language appearing where no English belongs. Every
phrase below is therefore English and word-bounded, and `I can't` requires a
following verb so a surname cannot trip it.

MEASURED, not asserted (see tests):
    precision  6 / 6      every page flagged in 62,320 was a genuine refusal
    recall     4 / 4      on the known refusals, including both delivered ones
    adversarial 0 / 6     false positives on real register text

WHAT WAS BUILT AND THEN CUT, AND WHY
------------------------------------
Three further detectors -- truncation, near-empty, repetition-loop -- were
written, measured on 701157, and REMOVED. Each was wrong for this corpus in a
way that only real data revealed:

    truncated   flagged 75 of 382 pages. It fired on pages ending in a priest's
                signature, "O Vigr.o Manoel Luis dos Reis Caval.o", because the
                final token has an internal period -- which is how every
                abbreviation in these registers is written. 96% false positives.
                Genuine page-truncated entries are already handled correctly by
                the segmenter's `partial` flag and cross-page stitching.

    repetition  fired on formulaic pages. A folio carrying several marriages
                repeats "r.do parocho da freguezia de n." once per entry. In a
                register this formulaic, a repeated n-gram is the NORM.

    empty       fired on cover and flyleaf pages whose transcription is "1" or
                "1
1830-1865
1833". Those are correct transcriptions of pages
                with nothing on them, and the router already classifies them.

A gate that flags one page in five trains people to ignore it, which is worse
than no gate. Only the detector with measured precision ships.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# English assistant-speak. Word-bounded; `I can't`/`I cannot` require a verb.
_REFUSAL = re.compile(
    r"(?:^|[\s.,;:\"'(])(?:"
    r"I'm sorry|I am sorry|I apologi[sz]e"
    r"|I cannot (?:transcribe|fulfill|provide|assist|complete|read|process|help)"
    r"|I can't (?:transcribe|fulfill|provide|assist|complete|read|process|help)"
    r"|I am unable to|I'm unable to|I'm not able to|I am not able to"
    r"|as an AI|I am an AI|I'm an AI|large language model"
    r"|programmed to be|helpful and harmless"
    r"|unable to (?:transcribe|process|read) (?:the|this)"
    r"|cannot fulfill this request"
    r"|(?:the )?(?:image|handwriting) is (?:too )?(?:blurry|illegible|faded|unclear)"
    r")", re.I)

# Scribal illegibility markers are NOT failures -- a good transcription of a
# damaged folio is full of them. Listed so nobody adds them to _REFUSAL later.
LEGITIMATE_GAPS = ("[torn]", "[illegible]", "[roto]", "[ilegible]", "[...]")



def check_page(text: Optional[str], page_id: str = "") -> Dict[str, Any]:
    """Reason codes for one page. `ok` False means DO NOT segment this."""
    t = (text or "").strip()
    problems: List[Dict[str, Any]] = []
    m = _REFUSAL.search(t)
    if m:
        problems.append({"code": "refusal", "detail": m.group(0).strip(),
                         "context": t[max(0, m.start() - 60):m.start() + 120]})
    return {"page": page_id, "ok": not problems, "problems": problems,
            "codes": [p["code"] for p in problems]}


def check_volume(volume: Any) -> Dict[str, Any]:
    """Every page of an Archivault volume export.

    Returns the failures and a clean/total count. The caller decides what to do
    -- this never edits or drops anything, because a page we cannot read is a
    fact to report, not a record to quietly discard.
    """
    pages = volume if isinstance(volume, list) else (
        volume.get("pages") or volume.get("entries") or [])
    failed, total = [], 0
    for p in pages:
        if not isinstance(p, dict):
            continue
        total += 1
        pid = str(p.get("file") or p.get("id") or p.get("image") or total)
        text = p.get("transcription") or p.get("text") or p.get("text_faithful")
        res = check_page(text, pid)
        if not res["ok"]:
            failed.append(res)
    return {"pages": total, "failed": len(failed),
            "rate": round(len(failed) / total, 6) if total else 0.0,
            "by_code": {"refusal": sum(1 for f in failed if "refusal" in f["codes"])},
            "failures": failed}


def format_report(report: Dict[str, Any], top: int = 12) -> str:
    lines = [f"transcription integrity: {report['pages'] - report['failed']:,}"
             f"/{report['pages']:,} pages complete"]
    if not report["failed"]:
        return lines[0] + "  (no problems)"
    lines.append(f"  {report['failed']} flagged ({100*report['rate']:.3f}%): "
                 f"{ {k: v for k, v in report['by_code'].items() if v} }")
    for f in report["failures"][:top]:
        first = f["problems"][0]
        lines.append(f"    {f['page']:24s} {first['code']:10s} "
                     f"{str(first.get('detail'))[:60]!r}")
    if report["failed"] > top:
        lines.append(f"    ... and {report['failed'] - top} more")
    lines.append("  A flagged page is NOT a hard page. Its text is not the "
                 "manuscript, so anything derived from it is invented.")
    return "\n".join(lines)
