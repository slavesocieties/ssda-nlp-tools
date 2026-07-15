"""Gold-set evaluation harness for SSDA entity extraction.

Scores predicted `data = {people, events}` against a hand-labeled gold set and
reports precision / recall / F1 per dimension:

  * people      — did we find the right people (name-aligned)?
  * attributes  — per-field accuracy on matched people (occupation, phenotype, ...)
  * events      — type + date + principals
  * relationships — directed typed edges over aligned people

The hard part is that predicted person IDs (P01…) are arbitrary and differ from
gold. We therefore align people by name *first*, then map every id-referencing
field (event principals, relationship endpoints) through that alignment into a
shared gold-name space before comparing. No LLM calls: this scores existing JSON.

Two input shapes are accepted transparently (see load_entries):
  * training-data:  {"examples": [{"entry","raw","normalized","data"}, ...]}
  * volume record:  {"entries": [{"id","raw","normalized","data"}, ...]}
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .textmatch import (
    greedy_align, name_similarity, normalize_name, prf, strip_accents, sum_prf,
)

PERSON_ATTRS = ["occupation", "phenotype", "free", "origin", "ethnicity",
                "age", "legitimate", "rank", "titles"]


# --------------------------------------------------------------------------- #
# loading / value normalization
# --------------------------------------------------------------------------- #

def load_entries(source: Any) -> List[Dict[str, Any]]:
    """Return a list of normalized entries: {id, text, people, events}."""
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            source = json.load(f)

    if isinstance(source, dict) and "examples" in source:
        rows = source["examples"]
        id_key = "entry"
    elif isinstance(source, dict) and "entries" in source:
        rows = source["entries"]
        id_key = "id"
    elif isinstance(source, list):
        rows = source
        id_key = "id"
    else:
        raise ValueError("Unrecognized input: expected {'examples'|'entries'} or a list.")

    out = []
    for r in rows:
        data = r.get("data") or {}
        out.append({
            "id": str(r.get(id_key, "")),
            "text": (r.get("normalized") or r.get("raw") or "").strip(),
            "people": data.get("people", []) or [],
            "events": data.get("events", []) or [],
        })
    return out


def norm_value(v: Any) -> Optional[str]:
    """Normalize an attribute value for comparison (case/accent/bool-insensitive)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        toks = sorted(norm_value(x) for x in v if x is not None and str(x).strip())
        return "|".join(t for t in toks if t) or None
    s = strip_accents(str(v)).lower().strip()
    if s in ("", "null", "none", "n/a"):
        return None
    if s in ("true", "false"):
        return s
    return s


# --------------------------------------------------------------------------- #
# entry alignment
# --------------------------------------------------------------------------- #

def align_entries(gold: List[dict], pred: List[dict], text_threshold: float = 0.6):
    """Match gold entries to pred entries by id, falling back to text similarity."""
    pred_by_id = {e["id"]: i for i, e in enumerate(pred) if e["id"]}
    matches, used_pred = [], set()

    for gi, g in enumerate(gold):
        pi = pred_by_id.get(g["id"])
        if pi is not None and pi not in used_pred:
            matches.append((gi, pi))
            used_pred.add(pi)

    # text fallback for anything still unmatched (handles different segmentation ids)
    rem_g = [gi for gi in range(len(gold)) if gi not in {m[0] for m in matches}]
    rem_p = [pi for pi in range(len(pred)) if pi not in used_pred]
    if rem_g and rem_p:
        scored = []
        for gi in rem_g:
            for pi in rem_p:
                from difflib import SequenceMatcher
                s = SequenceMatcher(None, gold[gi]["text"][:400].lower(),
                                    pred[pi]["text"][:400].lower()).ratio()
                if s >= text_threshold:
                    scored.append((s, gi, pi))
        scored.sort(reverse=True)
        ug, up = set(), set()
        for s, gi, pi in scored:
            if gi in ug or pi in up:
                continue
            matches.append((gi, pi))
            ug.add(gi); up.add(pi); used_pred.add(pi)

    unmatched_g = [gi for gi in range(len(gold)) if gi not in {m[0] for m in matches}]
    unmatched_p = [pi for pi in range(len(pred)) if pi not in used_pred]
    return matches, unmatched_g, unmatched_p


# --------------------------------------------------------------------------- #
# per-entry scoring
# --------------------------------------------------------------------------- #

def _canonical_map(people: List[dict], id_to_goldname: Dict[str, str]) -> Dict[str, str]:
    """person_id -> canonical name (aligned gold name if known, else pred:own-name)."""
    out = {}
    for p in people:
        pid = str(p.get("id", ""))
        if pid in id_to_goldname:
            out[pid] = id_to_goldname[pid]
        else:
            out[pid] = "pred:" + normalize_name(p.get("name"))
    return out


def score_entry(gold: dict, pred: dict, name_threshold: float = 0.72) -> Dict[str, Any]:
    gp, pp = gold["people"], pred["people"]
    matches, ug, up = greedy_align(gp, pp, key="name", threshold=name_threshold)

    # ---- people detection ----
    people_conf = prf(tp=len(matches), fp=len(up), fn=len(ug))

    # ---- id -> canonical gold-name maps (for events + relationships) ----
    gold_id_name = {str(gp[gi].get("id")): normalize_name(gp[gi].get("name")) for gi in range(len(gp))}
    pred_id_to_goldname = {str(pp[pi].get("id")): normalize_name(gp[gi].get("name"))
                           for gi, pi, _ in matches}
    gcanon = {str(p.get("id")): normalize_name(p.get("name")) for p in gp}
    pcanon = _canonical_map(pp, pred_id_to_goldname)

    # ---- attributes on matched people ----
    attr_stats = {a: {"gold": 0, "correct": 0, "pred": 0, "halluc": 0} for a in PERSON_ATTRS}
    attr_errors = []
    for gi, pi, _ in matches:
        g, p = gp[gi], pp[pi]
        for a in PERSON_ATTRS:
            gv, pv = norm_value(g.get(a)), norm_value(p.get(a))
            if gv is not None:
                attr_stats[a]["gold"] += 1
            if pv is not None:
                attr_stats[a]["pred"] += 1
            if gv is not None and pv is not None and gv == pv:
                attr_stats[a]["correct"] += 1
            elif gv is None and pv is not None:
                attr_stats[a]["halluc"] += 1
            if gv is not None and gv != pv:
                attr_errors.append({"person": g.get("name"), "attr": a,
                                    "gold": gv, "pred": pv})

    # ---- events (type + date + principals mapped to canonical names) ----
    def ev_repr(e, canon):
        principals = frozenset(canon.get(str(x), "?") for x in e.get("principals", []))
        return (norm_value(e.get("type")), principals, norm_value(e.get("date")))

    g_events = [ev_repr(e, gcanon) for e in gold["events"]]
    p_events = [ev_repr(e, pcanon) for e in pred["events"]]

    def ev_score(ga, pa):  # type must match; reward principal overlap + date
        if ga[0] != pa[0]:
            return 0.0
        s = 0.5
        if ga[1] and (ga[1] & pa[1]):
            s += 0.3 * len(ga[1] & pa[1]) / max(len(ga[1] | pa[1]), 1)
        if ga[2] and ga[2] == pa[2]:
            s += 0.2
        return s

    ge = [{"type": t, "prin": pr, "date": d} for (t, pr, d) in g_events]
    pe = [{"type": t, "prin": pr, "date": d} for (t, pr, d) in p_events]
    ev_matches, ev_ug, ev_up = greedy_align(
        ge, pe, threshold=0.5,
        score_fn=lambda a, b: ev_score((a["type"], a["prin"], a["date"]),
                                       (b["type"], b["prin"], b["date"])))
    events_conf = prf(tp=len(ev_matches), fp=len(ev_up), fn=len(ev_ug))
    date_gold = date_ok = 0
    for gi, pi, _ in ev_matches:
        if ge[gi]["date"] is not None:
            date_gold += 1
            if ge[gi]["date"] == pe[pi]["date"]:
                date_ok += 1

    # ---- relationships: directed typed edges over canonical names ----
    def rel_edges(people, canon):
        edges = set()
        for p in people:
            subj = canon.get(str(p.get("id")))
            for r in p.get("relationships", []) or []:
                if not isinstance(r, dict):
                    continue
                obj = canon.get(str(r.get("related_person")))
                rt = norm_value(r.get("relationship_type"))
                if subj and obj and rt:
                    edges.add((subj, rt, obj))
        return edges

    g_edges = rel_edges(gp, gcanon)
    p_edges = rel_edges(pp, pcanon)
    rel_tp = len(g_edges & p_edges)
    rel_conf = prf(tp=rel_tp, fp=len(p_edges - g_edges), fn=len(g_edges - p_edges))

    return {
        "entry": gold["id"] or pred["id"],
        "people": people_conf,
        "events": events_conf,
        "relationships": rel_conf,
        "date_on_matched_events": {"gold": date_gold, "correct": date_ok},
        "attr_stats": attr_stats,
        "attr_errors": attr_errors,
        "missed_people": [gp[gi].get("name") for gi in ug],
        "spurious_people": [pp[pi].get("name") for pi in up],
    }


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #

def evaluate(gold_source: Any, pred_source: Any, name_threshold: float = 0.72) -> Dict[str, Any]:
    gold = load_entries(gold_source)
    pred = load_entries(pred_source)
    ent_matches, ent_ug, ent_up = align_entries(gold, pred)

    per_entry = [score_entry(gold[gi], pred[pi], name_threshold) for gi, pi in ent_matches]

    # micro-average the confusion triples across entries
    def micro(dim):
        return sum_prf([e[dim] for e in per_entry]) if per_entry else prf(0, 0, 0)

    # attribute rollup
    attr_roll = {a: {"gold": 0, "correct": 0, "pred": 0, "halluc": 0} for a in PERSON_ATTRS}
    for e in per_entry:
        for a in PERSON_ATTRS:
            for k in attr_roll[a]:
                attr_roll[a][k] += e["attr_stats"][a][k]
    attr_report = {}
    for a, s in attr_roll.items():
        acc = s["correct"] / s["gold"] if s["gold"] else None
        halluc_rate = s["halluc"] / s["pred"] if s["pred"] else None
        attr_report[a] = {
            "gold_count": s["gold"],
            "accuracy": round(acc, 4) if acc is not None else None,
            "hallucination_rate": round(halluc_rate, 4) if halluc_rate is not None else None,
        }

    dg = sum(e["date_on_matched_events"]["gold"] for e in per_entry)
    dc = sum(e["date_on_matched_events"]["correct"] for e in per_entry)

    return {
        "entries": {
            "gold": len(gold), "pred": len(pred), "aligned": len(ent_matches),
            "unaligned_gold": len(ent_ug), "unaligned_pred": len(ent_up),
        },
        "people": micro("people"),
        "events": micro("events"),
        "relationships": micro("relationships"),
        "date_accuracy_on_matched_events": round(dc / dg, 4) if dg else None,
        "attributes": attr_report,
        "per_entry": per_entry,
    }


def format_report(report: Dict[str, Any], show_errors: bool = False) -> str:
    e = report["entries"]
    lines = []
    lines.append("=" * 66)
    lines.append("SSDA extraction evaluation")
    lines.append("=" * 66)
    lines.append(f"entries: {e['aligned']} aligned "
                 f"({e['gold']} gold, {e['pred']} pred, "
                 f"{e['unaligned_gold']} gold unmatched, {e['unaligned_pred']} pred unmatched)")
    lines.append("")
    lines.append(f"{'dimension':<16}{'precision':>11}{'recall':>9}{'f1':>8}"
                 f"{'tp':>6}{'fp':>5}{'fn':>5}")
    lines.append("-" * 66)
    for dim in ("people", "events", "relationships"):
        d = report[dim]
        lines.append(f"{dim:<16}{d['precision']:>11.3f}{d['recall']:>9.3f}{d['f1']:>8.3f}"
                     f"{d['tp']:>6}{d['fp']:>5}{d['fn']:>5}")
    da = report["date_accuracy_on_matched_events"]
    lines.append("")
    lines.append(f"date accuracy on matched events: {da if da is not None else 'n/a'}")
    lines.append("")
    lines.append("per-attribute (accuracy on matched people | hallucination rate):")
    for a, s in report["attributes"].items():
        if s["gold_count"] == 0 and s["hallucination_rate"] in (None, 0):
            continue
        acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "  -  "
        hal = f"{s['hallucination_rate']:.3f}" if s["hallucination_rate"] is not None else "  -  "
        lines.append(f"  {a:<12} acc={acc}  halluc={hal}  (gold n={s['gold_count']})")

    if show_errors:
        lines.append("")
        lines.append("error analysis:")
        for pe in report["per_entry"]:
            if pe["missed_people"] or pe["spurious_people"] or pe["attr_errors"]:
                lines.append(f"  [{pe['entry']}]")
                if pe["missed_people"]:
                    lines.append(f"     missed people   : {pe['missed_people']}")
                if pe["spurious_people"]:
                    lines.append(f"     spurious people : {pe['spurious_people']}")
                for ae in pe["attr_errors"]:
                    lines.append(f"     attr {ae['attr']} of {ae['person']!r}: "
                                 f"gold={ae['gold']!r} pred={ae['pred']!r}")
    lines.append("=" * 66)
    return "\n".join(lines)
