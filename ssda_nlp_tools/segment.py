"""Deterministic entry segmentation of Archivault page transcriptions (Task 3).

Parses per-image transcriptions into individual sacramental entries, matching
the paired examples' format:

    {"image": "<file>.jpg",
     "entries": [{"id": "<stem>-NN", "text": "...", "partial": false}, ...]}

Design: a line-classifying STATE MACHINE, not a bare regex split. Every line is
tagged (JUNK / OPENER / CLOSER / SIGNATURE / TEXT) and an opener only starts a
new entry when the machine is between entries — or, mid-entry, when the opener
is *strong* (full date formula) and the current entry already looks closed or
long. That is what stops "nascida a vinte e dous de Novembro" (a birth date
INSIDE an entry) from causing a false split.

Multilingual by construction (the gold pairs are Portuguese; volume 239746 is
Spanish): month names, weekday openers, and closing formulas for both languages.

Cost: $0 per image — pure regex/logic. Per-entry confidence is emitted so a
caller can route only low-confidence pages to an LLM fallback (see cost.py).
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# lexicons (Spanish + Portuguese)
# --------------------------------------------------------------------------- #

_MONTHS = (r"Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Se[pt]t?iembre|"
           r"Octubre|Noviembre|Diciembre|"
           r"Janeiro|Fevereiro|Mar[cç]o|Maio|Junho|Julho|Setembro|Outubro|"
           r"Novembro|Dezembro")
# b/v swaps are the canonical Spanish scribal variation ("Juebes", "Savado").
# The trailing s is optional for the same reason: these scribes drop it, and
# 15834 folio 0081 opens "Marte Veinte y quatro de Mayo de mill Sete Sientos".
# That page holds three records and the segmenter found ZERO, because with no
# opener anywhere the whole folio became one leading fragment. Dropping the s
# costs nothing in precision here -- `_OPENER` still requires a full "… de
# <MONTH>" date formula after the keyword, so a bare "Marte" or "Lune" in body
# text cannot open an entry on its own.
_WEEKDAYS = (r"Lunes?|Martes?|Mi[eé]rcoles?|Jue[bv]es?|[BV]iernes?|S[aá][bv]ados?|"
             r"Domingos?|Segunda|Ter[cç]a|Quarta|Quinta|Sexta")

# opener keywords: weekday-led ("Lunes, dia…"), "En/Em/Aos/Ao…", and the very
# common Spanish "El día…" / "La día…" / "El primero de…" forms used by whole
# dioceses (e.g. Cienfuegos vol 176899). "El/La día" is only treated as an entry
# start via the same year-strength / between-entry guards, so an in-body "nació
# el día veinte…" does not trigger a false split.
_OPENER_KW = (rf"(?:{_WEEKDAYS})\b[.,]?\s*(?:d[ií]a\b)?|"
              r"[EL][la]\s+d[ií]a|El\s+primero|En|Em|Aos|Ao|A\s?os|Á\s?os")

# strong opener: full date formula near line start (allows a short margin prefix)
_OPENER = re.compile(
    rf"^(?P<pre>[^\n]{{0,24}}?\s)??"
    rf"(?P<kw>{_OPENER_KW})"
    rf"[\s,]{{0,3}}"
    rf"(?P<body>(?:la\s+ciudad[^\n]{{0,40}}?|el\s+d[ií]a\s+)?[\w\s,.yeéí]{{0,70}}?)"
    rf"\s+de\s+(?:{_MONTHS})\b", re.IGNORECASE)

# Extra strength markers inside an opener line (spelled-out or digit year).
# "demil" is not a typo on our side: the transcriber frequently runs "de mil"
# together ("En dies y seis de Maio demil sett.os…"), which a bare \bmil\b misses
# — and since a weak opener never splits mid-entry, that silently swallowed whole
# records in 18th-c. volumes. Accept the concatenated form explicitly.
#
# `mil+` for the same reason, found the same way and measured against SSDA's hand
# transcriptions rather than guessed. The 18th-century scribes write the thousand
# as "mill": across volumes 15834/1795/419324 it is 371 occurrences against 1,604
# of "mil", so 19% of all year formulas were invisible to this pattern. 275
# opener lines carry year evidence ONLY in that spelling. Same failure mode as
# "demil" above — a weak opener never splits, so whole records were swallowed.
_YEARISH = re.compile(r"\b(?:de)?mil+\b|\b1[5-9]\d\d\b", re.IGNORECASE)

# loose opener: same formula shape but the month word may be TRUNCATED by the
# line wrap ("...de Noviem" / "bre de Mil..."). Only accepted with corroborating
# year/weekday evidence in the two-line probe (see classify_line).
_OPENER_LOOSE = re.compile(
    rf"^(?P<pre>[^\n]{{0,24}}?\s)??"
    rf"(?P<kw>{_OPENER_KW})"
    rf"[\s,]{{0,3}}"
    rf"(?P<body>(?:la\s+ciudad[^\n]{{0,40}}?|el\s+d[ií]a\s+)?[\w\s,.yeéí]{{0,70}}?)"
    rf"\s+de\s+[A-Za-zÀ-ÿ]{{3,12}}\b", re.IGNORECASE)

_CLOSER = re.compile(
    r"fi[zs]\s+este\s+as?s?ent|que\s+assign?[oe]i?\b|assinei\b|lo\s+firm[eéoó]|firmamos|"
    r"y\s+lo\s+firm|en\s+dicho\s+dia,?\s+mes,?|de\s+q\s+fiz|"
    r"mandei\s+fazer\s+este\s+assent|para\s+que\s+conste|\bconste\.|"
    r"obligaciones\s+y\s+parentesco\s+espiritual|parentesco\s+y\s+(?:demas\s+)?obligaciones",
    re.IGNORECASE)

# Officiant sign-off lines. Deliberately matches the ROLE prefix rather than any
# particular cleric: this used to also hardcode "O'Reilly|Hassett", the two
# priests of St. Augustine vol 239746, which is pure overfitting to the volume we
# tuned on. Removing them changes no boundary and no partial flag across all 5
# gold fixtures and all 5 Text data volumes — the role prefixes already cover
# those signatures — so the names bought nothing and only misled.
_SIGNATURE = re.compile(
    r"^\s*(?:O\s+Vig(?:ari)?[or]?\.?[oa]?\b|El\s+(?:Cura|P\.?e?\b)|Fr(?:ay|\.)\s|"
    r"Don\s+\w)", re.IGNORECASE)

_JUNK_LINE = re.compile(
    r"^\s*(?:\d{3,4}\.?|p[aá]g[.:]?\s*\d+\.{0,2}|folio\s*\d+|fol\.?\s*\d+\.?|"
    rf"(?:{_MONTHS})\s*(?:nada)?\.?|nada\.?|\d{{1,3}}\.{{0,2}}|"   # "47.." margin nums
    rf"[+=]\s*(?:{_WEEKDAYS})[.,]?|"          # stray "+ Miercoles" margin mark
    r"[=+][\w,.\s]{0,10})\s*$", re.IGNORECASE)  # catchword/reclamo: "=ma", "=ceno"

# minimum characters for a fragment to be a real entry/continuation; below this
# it is a catchword or margin overflow and goes to dropped_fragments
MIN_FRAGMENT = 25

# short margin-name line: 1-3 capitalized tokens (a name column fragment)
_MARGIN_NAME = re.compile(
    r"^\s*(?:[A-ZÁÉÍÓÚÑ][\w'’.\-]{1,15}[.,]?\s*){1,3}$")


def _strip_accents_lower(s: str) -> str:
    n = unicodedata.normalize("NFKD", s)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


# --------------------------------------------------------------------------- #
# line classification
# --------------------------------------------------------------------------- #

def classify_line(line: str, state: str, lookahead: str = "") -> str:
    """Tag one line: junk | opener | opener_strong | closer | signature | text.

    `lookahead` = the following line; the opener test runs on the two lines
    JOINED, because the date formula is frequently broken mid-word at the line
    wrap ("...de Noviem" / "bre de Mil...") and would otherwise be invisible.
    """
    s = line.strip()
    if not s:
        return "blank"
    if _JUNK_LINE.match(s):
        return "junk"
    probe = (s + " " + lookahead.strip())[:220] if lookahead else s
    weekday_led = bool(re.match(rf"^(?:{_WEEKDAYS})\b", s, re.IGNORECASE))
    m = _OPENER.match(s)
    if not m:
        # look ahead across the wrap, but ONLY if the keyword itself begins on
        # THIS line — otherwise a margin name right above a "Domingo, dia..."
        # line would wrongly become the entry start ("Friay." + lookahead)
        pm = _OPENER.match(join_lines([s, lookahead.strip()])[:220])
        if pm and pm.start("kw") < len(s):
            m = pm
    if not m:
        # month word truncated at the wrap — accept only with year/weekday proof
        lm = _OPENER_LOOSE.match(s)
        if lm and (weekday_led or _YEARISH.search(probe)):
            m = lm
    if m:
        # "El día…" / "La día…" are common IN-BODY ("nació el día quinze de
        # Junio…"), unlike a weekday which almost always begins a record. So an
        # El/La-día opener only counts when it carries a year ("…de mil…"); the
        # weekday and En/Aos openers stay weak-eligible.
        kw = (m.group("kw") or "").lower()
        el_dia_family = bool(re.match(r"[el]l?a?\s+d[ií]a|el\s+primero", kw)) \
            and not weekday_led
        has_year = bool(_YEARISH.search(probe))
        if el_dia_family and not has_year:
            return "text"
        if has_year or weekday_led:
            return "opener_strong"
        return "opener"
    if _CLOSER.search(s):
        return "closer"
    if _SIGNATURE.match(s) and len(s) <= 60:
        return "signature"
    if state == "between" and _MARGIN_NAME.match(s) and len(s) <= 28:
        return "junk"                     # margin name column between entries
    return "text"


# --------------------------------------------------------------------------- #
# joining lines into entry text
# --------------------------------------------------------------------------- #

def join_lines(lines: List[str]) -> str:
    """Re-flow entry lines: heal line-break continuation marks (- and =),
    keep single spaces. Both marks appear in the corpus: "Setecien/-tos" and
    "nom/=bre" mean the same broken word."""
    out = ""
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if not out:
            out = s
        elif s[0] in "-=":
            if out.endswith(("-", "=")):
                out = out[:-1]             # "Mig=" + "=uel" -> "Miguel"
            out += s[1:].lstrip()          # "Setecien"+"-tos", "nom"+"=bre"
        elif out.endswith("-"):
            out = out[:-1] + s             # "pre-" + "zente" -> "prezente"
        elif out.endswith("="):
            # end-"=" is BOTH a word-break ("Miguel O=/Reilly") and a stop
            # ("...y Año=" before the signature). Glue only when it looks like
            # a broken word: a 1-2 char stub or a lowercase continuation.
            stub = re.search(r"(?:^|[^\w])(\w{1,2})=$", out)   # whole final word 1-2 chars
            if stub or (s and s[0].islower()):
                out = out[:-1] + s
            else:
                out += " " + s
        else:
            out += " " + s
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------- #
# per-page segmentation (state machine)
# --------------------------------------------------------------------------- #

_FOLIO_REF = re.compile(r"\bf(?:ol)?\.?\s*\d{1,4}\s*\|?\s*$", re.IGNORECASE)

# Archivault API failures embedded verbatim in the corpus transcriptions
_TRANSCRIPTION_ERROR = re.compile(
    r"\[transcription failed|max retries reached|finish\s?reason[:.]", re.IGNORECASE)


# A markdown table row, and the |---|---| rule under its header.
_MD_ROW = re.compile(r"^\s*\|.*\|?\s*$")
_MD_SEP = re.compile(r"^\s*\|[\s\-:|]*\|?\s*$")

# Must match the `pre` group width in _OPENER: a longer prefix stops the line
# reading as an opener at all.
_MARGIN_PREFIX_MAX = 24


def strip_markdown_table(text: str) -> str:
    """Flatten a markdown table back into lines of register prose.

    These registers are laid out in two columns -- a margin name and the record
    body -- and Gemini often renders that faithfully as a markdown table:

        | Victoriano Mora | En esta Santa Yglesia de Santa Cruz de Lorica a |
        | Partida 8241.   | treinta de Mayo de mil ochocientos ochenta y    |

    which is a reasonable reading of the page and completely breaks the
    segmenter. `detect_page_type` counts lines beginning with "|" as evidence of
    an INDEX page, so a register in this shape is skipped whole: folios 0007,
    0013 and 0015 of 419324 each hold seven records and produced zero.

    Corpus-wide this is 3,622 of 62,320 pages (5.81%) across 123 of the 232
    volumes, and volume 439941 is 85.8% tables. Every one of those pages would
    have silently yielded nothing.

    Separator rules are dropped. Text with no table in it is returned unchanged.

    COLUMN-AWARE, and it has to be. Naively joining every row's cells left to
    right splices the margin column into the middle of the body prose, because
    the margin annotation wraps across the same rows the record does:

        | Braulia    | Aos desasseis dias do mez de Junho de mil oitocentos e ses- |
        | innocente  | senta e cinco encommendei e sepultou-se o cadaver           |

    joins to "...de mil oitocentos e ses- innocente senta e cinco...", breaking
    "sessenta" in half with a word that belongs to the margin. That damages the
    date, which is the one field a burial record cannot afford to lose.

    So for a two-column table the narrower column is treated as the margin and
    emitted on its OWN line, except on rows where the body cell opens a record --
    there it stays in front, which is exactly the margin-prefix shape
    `_strip_margin_prefix` already handles. A standalone margin line is then
    classified as junk by the existing `_MARGIN_NAME` rule, so nothing is lost
    and nothing is spliced.
    """
    lines = text.splitlines()
    rows = [ln for ln in lines if ln.strip() and _MD_ROW.match(ln)]
    if len(rows) < 3 or len(rows) < 0.5 * len([l for l in lines if l.strip()]):
        return text

    split = [[c.strip() for c in ln.strip().strip("|").split("|")]
             for ln in rows if not _MD_SEP.match(ln)]
    # Which column is the margin? The narrower one, by total text.
    margin_col = None
    if split and all(len(c) == 2 for c in split):
        w0 = sum(len(c[0]) for c in split)
        w1 = sum(len(c[1]) for c in split)
        if w0 and w1 and min(w0, w1) * 3 < max(w0, w1):
            margin_col = 0 if w0 < w1 else 1

    # Two columns: rebuild the body stream, and gather each record's margin
    # annotation onto the line that OPENS that record.
    #
    # Emitting a margin cell on its own line is not enough. `join_lines` heals
    # the manuscript's hyphenated line wraps by concatenation, so a lone
    # "innocente" sitting between "…oitocentos e ses-" and "senta e cinco"
    # rejoins as "sesinnocente senta" and the year is still destroyed. The
    # margin has to leave the body stream entirely.
    if margin_col is not None:
        out: List[str] = []
        pending: List[str] = []          # margin cells awaiting their opener
        body_lines: List[str] = []
        for ln in lines:
            if not ln.strip() or _MD_SEP.match(ln):
                continue
            if not _MD_ROW.match(ln):
                body_lines.append(ln.strip())
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) != 2:
                joined = " ".join(c for c in cells if c)
                if joined:
                    body_lines.append(joined)
                continue
            margin, body = cells[margin_col], cells[1 - margin_col]
            opens = bool(body) and classify_line(body, "between") in (
                "opener", "opener_strong")
            if opens:
                out.extend(body_lines)
                body_lines = []
                pending.append(margin)
                prefix = " ".join(p for p in pending if p).strip()
                pending = []
                # `_OPENER` allows a margin prefix of at most 24 characters, so
                # a long accumulated one does not read as a prefix -- it hides
                # the opener entirely and the record silently merges into its
                # predecessor. Measured: prefixing unconditionally dropped
                # 419324 from 80% to 60% exact folio counts. Over the limit, the
                # margin goes on its own line just BEFORE the opener, where it
                # is a clean boundary rather than a splice into running prose.
                if prefix and len(prefix) <= _MARGIN_PREFIX_MAX:
                    out.append(f"{prefix} {body}".strip())
                else:
                    if prefix:
                        out.append(prefix)
                    out.append(body)
                continue
            if margin:
                pending.append(margin)
            if body:
                body_lines.append(body)
        out.extend(body_lines)
        if pending:
            out.append(" ".join(p for p in pending if p).strip())
        return "\n".join(ln for ln in out if ln)

    out = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue
        if _MD_SEP.match(ln):
            continue
        if _MD_ROW.match(ln):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            joined = " ".join(c for c in cells if c)
            if joined:
                out.append(joined)
            continue
        out.append(ln)
    return "\n".join(out)


def detect_page_type(text: str) -> str:
    """'register' | 'index' | 'error' | 'blank'.

    Index pages (name -> folio tables) are not entries and need no LLM fallback
    — skip them. 'error' pages carry a verbatim Archivault API failure string
    ("[transcription failed: max retries reached]") and must be RE-TRANSCRIBED,
    not segmented.

    Runs on the table-flattened text, so that a REGISTER rendered as a markdown
    table is not mistaken for an index. A genuine index survives the flattening
    because what identifies it is the folio references and the short lines, not
    the pipes.
    """
    if _TRANSCRIPTION_ERROR.search(text):
        return "error"
    text = strip_markdown_table(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or len(text.strip()) < 120:
        return "blank"
    tableish = sum(1 for ln in lines if ln.lstrip().startswith("|") or _FOLIO_REF.search(ln))
    if len(lines) >= 5 and tableish / len(lines) >= 0.4:
        return "index"
    return "register"


def segment_page(text: str, image: str = "", min_split_len: int = 150) -> Dict[str, Any]:
    """Segment ONE page transcription into entries (page-independent, like the
    gold pairs: a trailing unfinished entry gets partial=true)."""
    stem = re.sub(r"\.(jpe?g|png|tiff?)$", "", image, flags=re.IGNORECASE) or "page"
    ptype = detect_page_type(text)
    # Flatten a markdown-table rendering of the two-column layout before the
    # line classifier sees it; a no-op on ordinary pages.
    text = strip_markdown_table(text)
    if ptype == "index":
        return {"image": image, "entries": [], "leading_fragment": None,
                "dropped_fragments": [], "page_type": "index", "confidence": 0.85}
    if ptype == "error":
        # verbatim API-failure text — the page needs RE-TRANSCRIPTION upstream
        return {"image": image, "entries": [], "leading_fragment": None,
                "dropped_fragments": [], "page_type": "error", "confidence": 0.0}
    lines = text.splitlines()

    entries: List[Dict[str, Any]] = []
    dropped: List[str] = []     # catchwords / margin overflow too short to be entries
    cur: List[str] = []
    cur_closed = False          # saw a closer in the current entry
    cur_signed = False          # saw the signature (entry is complete)
    state = "between"
    leading: List[str] = []     # text before the first opener (page-top continuation)

    def flush(partial: bool):
        nonlocal cur, cur_closed, cur_signed
        body = join_lines(cur)
        if body and len(body) >= MIN_FRAGMENT:
            entries.append({"id": f"{stem}-{len(entries) + 1:02d}",
                            "text": body, "partial": bool(partial)})
        elif body:
            dropped.append(body)
        cur, cur_closed, cur_signed = [], False, False

    def _strip_margin_prefix(line: str) -> str:
        """Drop a short margin-name prefix from an opener line — both gold pairs
        do this ('Pedro Aos vinte...' -> 'Aos vinte...', 'U. Inoc.e Em 3 de...'
        -> 'Em 3 de...')."""
        s = line.strip()
        m = _OPENER.match(s) or _OPENER_LOOSE.match(s)
        if m and m.group("pre"):
            return s[m.start("kw"):]
        return line

    for li, raw in enumerate(lines):
        nxt = lines[li + 1] if li + 1 < len(lines) else ""
        tag = classify_line(raw, state, lookahead=nxt)
        if tag in ("blank", "junk"):
            continue
        if tag in ("opener", "opener_strong"):
            raw = _strip_margin_prefix(raw)
            if state == "between":
                if cur:                      # previous entry fully closed
                    flush(partial=False)
                state = "inside"
                cur.append(raw)
                continue
            # inside an entry: split only on strong evidence. The entry we are
            # closing is followed by a NEW dated entry on the same page, so it
            # cannot be continuing anywhere — registers are sequential. Never
            # mark it partial (only a PAGE-FINAL unclosed entry can continue).
            if cur_signed or (tag == "opener_strong"
                              and (cur_closed or len(join_lines(cur)) >= min_split_len)):
                flush(partial=False)
                state = "inside"
                cur.append(raw)
            else:
                cur.append(raw)              # in-entry date mention -> body text
            continue
        if state == "between" and not entries and tag in ("text", "closer", "signature"):
            leading.append(raw)              # continuation of previous PAGE's entry
            if tag == "signature" or (tag == "closer" and _SIGNATURE.match(raw or "")):
                pass
            continue
        if state == "between" and tag in ("text",):
            # stray text between entries (margin block overflow) — keep with next
            cur.append(raw)
            continue
        # inside an entry
        cur.append(raw)
        if tag == "closer":
            cur_closed = True
        elif tag == "signature" and cur_closed:
            cur_signed = True
            state = "between"
            flush(partial=False)

    if cur:
        flush(partial=not (cur_closed or cur_signed))

    lead_text = join_lines(leading)
    short_lead = None
    if lead_text and len(lead_text) < MIN_FRAGMENT:
        # Usually a catchword/reclamo at the page top rather than a continuation,
        # so it does not become a leading_fragment. But it is sometimes the last
        # few words of a record spilling over ("y lo firmé. Fr. Juan"), and
        # dropping it silently LOSES archival text. Hand it to segment_volume,
        # which can see whether the previous page actually left a record open.
        short_lead = lead_text
        dropped.append(lead_text)
        lead_text = ""
    return {"image": image, "entries": entries,
            "leading_fragment": lead_text or None,
            "short_leading": short_lead,
            "dropped_fragments": dropped,
            "confidence": _page_confidence(entries, lead_text, text)}


def _page_confidence(entries: List[dict], leading: str, raw_text: str = "") -> float:
    """0..1: how much to trust this page's deterministic segmentation."""
    if not entries:
        # near-empty page = cover/title/blank — finding nothing IS the answer.
        # A text-rich page with NO anchored entry is only acceptable if its text
        # attaches to the previous page's unfinished entry; segment_volume bumps
        # the confidence when that attachment succeeds. Standalone, it's suspect.
        return 0.9 if len(raw_text.strip()) < 120 else 0.45
    score = 1.0
    for e in entries:
        n = len(e["text"])
        if n < 120 and not e["partial"]:
            score -= 0.15             # suspiciously short "complete" entry
        if n > 3500:
            score -= 0.2              # suspiciously long — probable missed split
    if leading and len(leading) > 2000:
        score -= 0.2                  # huge unanchored continuation
    return max(0.0, min(1.0, round(score, 3)))


# --------------------------------------------------------------------------- #
# volume mode: stitch cross-page partials
# --------------------------------------------------------------------------- #

def segment_volume(pages: List[Tuple[str, str]], min_split_len: int = 150) -> Dict[str, Any]:
    """pages = [(image_file, transcription), ...] in reading order.

    Returns {"per_image": [gold-style page dicts], "entries": merged volume-level
    entries (cross-page partials stitched), "low_confidence": [image names]}.
    """
    per_image = [segment_page(t, image=img, min_split_len=min_split_len)
                 for img, t in pages]

    merged: List[Dict[str, Any]] = []
    for pg in per_image:
        frag = pg.get("leading_fragment")
        # A sub-MIN_FRAGMENT page-top scrap counts as a continuation ONLY when
        # the previous page genuinely left a record open. With something to
        # attach to it is a spillover; with nothing, it stays a catchword and is
        # never promoted into an orphan record of its own.
        if not frag and pg.get("short_leading") and merged \
                and merged[-1]["partial"] and not merged[-1].get("orphan"):
            frag = pg["short_leading"]
        if frag and merged and merged[-1]["partial"] and not merged[-1].get("orphan"):
            prev = merged[-1]
            prev["text"] = join_lines([prev["text"], frag])
            prev["partial"] = not (_CLOSER.search(frag) or _SIGNATURE.search(frag))
            prev.setdefault("source_images", []).append(pg["image"])
            # the fragment found its home — this page's segmentation is sound
            if pg["confidence"] < 0.85 and not pg["entries"]:
                pg["confidence"] = 0.85
        elif frag:
            # Nothing to attach to. Usually a continuation from a page we don't
            # have — but NOT always: some registers open a record with no date at
            # all (740018-0006-01 starts "El R.do P.e fr. Sebastian…"), and the
            # manual gold counts those as complete records. "partial" must mean
            # "runs off the page", so decide it from the evidence: a fragment
            # carrying its own closing formula/signature is complete.
            complete = bool(_CLOSER.search(frag) or _SIGNATURE.search(frag))
            merged.append({"id": f"{_stem(pg['image'])}-00", "text": frag,
                           "partial": not complete, "orphan": True,
                           "source_images": [pg["image"]]})
        for e in pg["entries"]:
            merged.append({**e, "source_images": [pg["image"]]})

    _resolve_partials(merged, per_image)

    errors = [pg["image"] for pg in per_image if pg.get("page_type") == "error"]
    low = [pg["image"] for pg in per_image
           if pg["confidence"] < 0.7 and pg.get("page_type") != "error"]
    return {"per_image": per_image, "entries": merged, "low_confidence": low,
            "error_pages": errors,
            "stats": {"pages": len(pages), "entries": len(merged),
                      "cross_page": sum(1 for e in merged
                                        if len(e.get("source_images", [])) > 1),
                      "still_partial": sum(1 for e in merged if e["partial"]),
                      "low_confidence_pages": len(low),
                      "error_pages": len(errors)}}


def _resolve_partials(merged: List[Dict[str, Any]],
                      per_image: List[Dict[str, Any]]) -> None:
    """Decide `partial` from PAGE POSITION, not from the closing-formula lexicon.

    `partial` is supposed to mean "this record runs off the page". But
    segment_page is page-local: it can only guess that from _CLOSER/_SIGNATURE,
    and those lexicons are harvested per register format. On a volume whose
    closing formula we do not know, EVERY page-final record looks unclosed and
    is delivered as truncated — even though the next page proves it ended. That
    is a lexicon-coverage artifact, not a real truncation, and it scales with
    the number of page breaks rather than with real run-ons.

    The next page settles it: if the following register page opens its own
    record and contributed no continuation fragment, then nothing carried over
    and the record is complete. Only these stay flagged:
      * records ending on the volume's last page (no next page to confirm)
      * records followed by an index/error page (unreadable -> cannot confirm)
      * records still genuinely continuing (next page gave a leading fragment)
    """
    pos = {pg["image"]: i for i, pg in enumerate(per_image)}

    def _page_of(entry, end: bool):
        imgs = entry.get("source_images") or []
        return pos.get(imgs[-1] if end else imgs[0]) if imgs else None

    for k, e in enumerate(merged):
        if not e.get("partial"):
            continue
        i = _page_of(e, end=True)
        if i is None:
            continue
        # (a) another record starts on the same page after this one. Registers
        # are sequential, so this record demonstrably closed mid-page — the same
        # argument segment_page already uses for a mid-page opener split.
        if k + 1 < len(merged):
            j = _page_of(merged[k + 1], end=False)
            if j is not None and j <= i:
                e["partial"] = False
                continue
        # (b) it really did run to the foot of page i — let page i+1 settle it.
        if i + 1 >= len(per_image):
            continue                       # ends the volume — genuinely unconfirmed
        nxt = per_image[i + 1]
        if nxt.get("page_type") in ("index", "error"):
            continue                       # can't be read — keep the flag honest
        if nxt.get("leading_fragment"):
            continue                       # really does continue
        if not nxt.get("entries"):
            continue                       # blank/unsegmented — proves nothing
        e["partial"] = False               # next page starts clean -> record ended


def _stem(image: str) -> str:
    return re.sub(r"\.(jpe?g|png|tiff?)$", "", image, flags=re.IGNORECASE)


def to_canonical(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert volume-mode entries to the CONFIRMED canonical schema (Daniel,
    2026-07-16): a flat list of {"id", "text", "images": [...]}, where the id is
    inferred from the image list — "<stem of first image>-NN", numbered in
    reading order per starting image (matches the manual gold exactly, e.g.
    740018-0006-01). "partial" is included only when true: incomplete records
    are kept and flagged, never dropped (also per Daniel), but complete records
    don't carry dead fields.
    """
    counters: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for e in entries:
        images = e.get("source_images") or []
        first = _stem(images[0]) if images else _stem(str(e.get("id", "entry")))
        counters[first] = counters.get(first, 0) + 1
        row: Dict[str, Any] = {"id": f"{first}-{counters[first]:02d}",
                               "text": e["text"],
                               "images": list(images)}
        if e.get("partial"):
            row["partial"] = True
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# input loading (all three observed shapes)
# --------------------------------------------------------------------------- #

def load_pages(source: Any) -> List[Tuple[str, str]]:
    """Accepts: Archivault volume JSON ([{images:[{file,transcription}]}]),
    a single {file,transcription} dict, or the .md pair format."""
    if isinstance(source, str) and source.lower().endswith(".md"):
        return _load_md(source)
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            source = json.load(f)
    if isinstance(source, dict) and "transcription" in source:
        return [(source.get("file", "page.jpg"), source["transcription"])]
    items = source if isinstance(source, list) else [source]
    pages: List[Tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "transcription" in item:            # flat corpus format: [{file, transcription}]
            t = item.get("transcription") or ""
            if t.strip():
                pages.append((item.get("file", f"page{len(pages)}.jpg"), t))
            continue
        for img in item.get("images", []) or []:
            t = img.get("transcription") or ""
            if t.strip():
                pages.append((img.get("file", f"page{len(pages)}.jpg"), t))
    return pages


def _load_md(path: str) -> List[Tuple[str, str]]:
    text = open(path, "r", encoding="utf-8").read()
    pages = []
    for m in re.finditer(r"###\s+(\S+)(.*?)(?=###\s+\S+|\Z)", text, re.S):
        fname, block = m.group(1), m.group(2)
        tm = re.search(r"\*\*transcription\*\*:\s*\n(.*?)(?=\n-\s+\*\*|\Z)", block, re.S)
        if tm:
            body = "\n".join(ln.strip() for ln in tm.group(1).splitlines())
            pages.append((fname, body.strip()))
    return pages
