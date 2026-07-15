"""Volume QA report — data-quality checks that need no gold labels.

Catches the failure modes we have actually observed in real outputs, so problems
surface per-volume at scale instead of silently polluting the database:

  * duplicate entries   — the LLM window-repair double-reports a record when the
                          overlapping windows transcribe it slightly differently
                          (its first-500-chars dedup misses them; we fuzzy-match)
  * chronology          — registers are chronological; a date that jumps backwards
                          or an impossible/out-of-era date means a bad extraction
  * dangling references — relationships/principals pointing at missing person ids
  * reciprocity         — one-directional relationship edges (count only)
  * event-shape rules   — baptism=1 principal, marriage=2
  * vocabulary drift    — value distributions for phenotype/ethnicity/origin
                          (surfaces 'negra' vs 'negro', Spanish vs English, ...)

Everything is pure JSON-in / report-out; no LLM, no network.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .fixes import RECIPROCAL_RELS

_WS = re.compile(r"\s+")


def _load(source: Any) -> dict:
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    return source


def _entries(vol: dict) -> List[dict]:
    return vol.get("entries") or vol.get("examples") or []


def _eid(e: dict) -> str:
    return str(e.get("entry") or e.get("id") or "")


def _text(e: dict) -> str:
    return _WS.sub(" ", (e.get("normalized") or e.get("raw") or "")).strip().lower()


def _parse_iso(date: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not date or not isinstance(date, str):
        return None
    m = re.match(r"^(\d{3,4})-(\d{1,2})-(\d{1,2})$", date.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


_MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 29: be lenient


def qa_volume(source: Any,
              dup_threshold: float = 0.75,
              year_range: Tuple[int, int] = (1500, 1900)) -> Dict[str, Any]:
    vol = _load(source)
    entries = _entries(vol)
    issues: List[dict] = []

    # ---- 1. near-duplicate entries (windowed: compare each entry to next 4) ----
    # Sacramental text is FORMULAIC: two different baptisms in the same family
    # can be >0.9 text-similar. So high text similarity alone is not enough —
    # the sacrament PRINCIPAL (the person baptized/buried) must also match.
    def _principal_name(e: dict) -> Optional[str]:
        data = e.get("data") or {}
        people = {str(p.get("id")): p.get("name") for p in data.get("people", []) or []}
        for ev in data.get("events", []) or []:
            if str(ev.get("type", "")).lower() in ("baptism", "burial"):
                pr = ev.get("principals") or []
                if pr:
                    return people.get(str(pr[0]))
        return None

    def _same_principal(na: Optional[str], nb: Optional[str]) -> Optional[bool]:
        if not na or not nb:
            return None                      # unknown — can't corroborate
        from .disambiguate import _third_party_same
        return _third_party_same(na, nb)

    texts = [(_eid(e), _text(e), _principal_name(e)) for e in entries]
    duplicates = []
    for i in range(len(texts)):
        for j in range(i + 1, min(i + 5, len(texts))):
            a, b = texts[i][1], texts[j][1]
            if not a or not b:
                continue
            # cheap length pre-filter before the quadratic matcher
            if abs(len(a) - len(b)) > 0.35 * max(len(a), len(b)):
                continue
            sim = SequenceMatcher(None, a[:800], b[:800]).ratio()
            if sim < dup_threshold:
                continue
            same = _same_principal(texts[i][2], texts[j][2])
            if same is False:
                continue      # different sacrament principals -> two real records,
                              # however formulaic-similar the wording is
            confidence = "confirmed" if same else "unconfirmed(no principal)"
            duplicates.append({"a": texts[i][0], "b": texts[j][0], "sim": round(sim, 3),
                               "principal": texts[i][2], "confidence": confidence})
            issues.append({"type": "duplicate_entry", "entry": texts[j][0],
                           "detail": f"{texts[j][0]} duplicates {texts[i][0]} "
                                     f"(sim {sim:.2f}, principal {texts[i][2]!r}, {confidence})"})

    # ---- 2. chronology over primary event dates ----
    dated = []
    for e in entries:
        for ev in (e.get("data") or {}).get("events", []) or []:
            d = _parse_iso(ev.get("date"))
            if d and str(ev.get("type", "")).lower() in ("baptism", "marriage", "burial"):
                dated.append((_eid(e), d))
                break
    chron_breaks = []
    for e in entries:
        for ev in (e.get("data") or {}).get("events", []) or []:
            d = _parse_iso(ev.get("date"))
            if d is None:
                continue
            y, mo, da = d
            if not (year_range[0] <= y <= year_range[1]) or not (1 <= mo <= 12) \
                    or not (1 <= da <= _MONTH_DAYS[mo - 1]):
                issues.append({"type": "impossible_date", "entry": _eid(e),
                               "detail": f"{ev.get('type')} date {ev.get('date')!r}"})
    for k in range(1, len(dated)):
        (e_prev, d_prev), (e_cur, d_cur) = dated[k - 1], dated[k]
        # registers run forward; allow same-day, flag > 30-day regressions
        days_prev = d_prev[0] * 372 + d_prev[1] * 31 + d_prev[2]
        days_cur = d_cur[0] * 372 + d_cur[1] * 31 + d_cur[2]
        if days_cur < days_prev - 30:
            chron_breaks.append({"prev_entry": e_prev, "entry": e_cur,
                                 "prev": "-".join(map(str, d_prev)),
                                 "cur": "-".join(map(str, d_cur))})
            issues.append({"type": "chronology_break", "entry": e_cur,
                           "detail": f"{'-'.join(map(str, d_cur))} after "
                                     f"{'-'.join(map(str, d_prev))} ({e_prev})"})

    # ---- 3. dangling refs / 4. reciprocity / 5. event shape ----
    recip_misses = 0
    for e in entries:
        data = e.get("data") or {}
        people = data.get("people", []) or []
        ids = {str(p.get("id")) for p in people}
        rel = {str(p.get("id")): {} for p in people}
        for p in people:
            for r in p.get("relationships", []) or []:
                if not isinstance(r, dict):
                    continue
                tgt = str(r.get("related_person"))
                if tgt not in ids:
                    issues.append({"type": "dangling_relationship", "entry": _eid(e),
                                   "detail": f"{p.get('id')} -> missing {tgt}"})
                else:
                    rel[str(p.get("id"))][tgt] = r.get("relationship_type")
        for a, targets in rel.items():
            for b, t in targets.items():
                expected = RECIPROCAL_RELS.get(t)
                if expected and rel.get(b, {}).get(a) != expected:
                    recip_misses += 1
        for ev in data.get("events", []) or []:
            et = str(ev.get("type", "")).lower()
            principals = ev.get("principals", []) or []
            for pid in principals:
                if str(pid) not in ids:
                    issues.append({"type": "dangling_principal", "entry": _eid(e),
                                   "detail": f"{et} -> missing {pid}"})
            if et == "baptism" and len(principals) != 1:
                issues.append({"type": "event_shape", "entry": _eid(e),
                               "detail": f"baptism with {len(principals)} principals"})
            if et == "marriage" and len(principals) != 2:
                issues.append({"type": "event_shape", "entry": _eid(e),
                               "detail": f"marriage with {len(principals)} principals"})

    # ---- 6. vocabulary distributions (drift detector) ----
    vocab: Dict[str, Counter] = {k: Counter() for k in ("phenotype", "ethnicity", "origin", "age")}
    empty_entries = []
    for e in entries:
        people = (e.get("data") or {}).get("people", []) or []
        if not people:
            empty_entries.append(_eid(e))
            issues.append({"type": "no_people", "entry": _eid(e), "detail": "entry has no people"})
        for p in people:
            for k in vocab:
                v = p.get(k)
                if v is not None and str(v).strip():
                    vocab[k][str(v).strip().lower()] += 1

    by_type = Counter(i["type"] for i in issues)
    return {
        "volume": vol.get("title") or vol.get("id"),
        "entries": len(entries),
        "issues": issues,
        "issues_by_type": dict(by_type),
        "duplicates": duplicates,
        "chronology_breaks": chron_breaks,
        "nonreciprocal_relationships": recip_misses,
        "vocabulary": {k: dict(c.most_common()) for k, c in vocab.items()},
    }


def format_qa(report: Dict[str, Any], max_rows: int = 12) -> str:
    lines = ["=" * 64, f"Volume QA — {report['volume']}  ({report['entries']} entries)",
             "=" * 64]
    bt = report["issues_by_type"]
    if not bt:
        lines.append("no issues found")
    else:
        lines.append("issues: " + ", ".join(f"{k}={v}" for k, v in sorted(bt.items())))
    if report["duplicates"]:
        lines.append("")
        lines.append("suspected duplicate entries (window-overlap re-transcriptions):")
        for d in report["duplicates"][:max_rows]:
            lines.append(f"  {d['a']} ~ {d['b']}  (sim {d['sim']}, "
                         f"principal {d.get('principal')!r}, {d.get('confidence')})")
    if report["chronology_breaks"]:
        lines.append("")
        lines.append("chronology breaks:")
        for c in report["chronology_breaks"][:max_rows]:
            lines.append(f"  {c['entry']}: {c['cur']} follows {c['prev']} ({c['prev_entry']})")
    if report["nonreciprocal_relationships"]:
        lines.append("")
        lines.append(f"non-reciprocal relationship edges: {report['nonreciprocal_relationships']}")
    lines.append("")
    lines.append("vocabulary (drift check):")
    for k, dist in report["vocabulary"].items():
        if dist:
            head = ", ".join(f"{v}×{c}" for v, c in list(dist.items())[:8])
            lines.append(f"  {k:<10} {head}")
    lines.append("=" * 64)
    return "\n".join(lines)
