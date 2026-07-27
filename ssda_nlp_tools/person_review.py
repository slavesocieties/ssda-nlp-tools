"""Per-person review surface — the reframing that makes the queue tractable.

Measured on the delivered corpus (see eval_data/disambiguation_scale_20260727.md):
contextual blocking cut the borderline queue from 795,713 pairs to 612,495, a
23% reduction. That is not the problem solved; 612,495 pair decisions is as
unreviewable as 796,000.

Grouping the SAME pairs by person changes the arithmetic instead of the count:
one identity carrying 40 candidates is 40 rows in a pair queue but one screen
here, and a reviewer answers "which of these are the same person as this one?"
once rather than forty times. Most identities have no candidate at all and never
need to be shown.

This is a pure regrouping of `disambiguate_volume()["review_queue"]`. It invents
no pairs and drops none: every pair appears under both of its people, and the
decisions it produces stay in the must/cannot form `run_review.py apply`
already consumes, so nothing downstream changes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def _key(side: Dict[str, Any]) -> str:
    return f"{side.get('entry')}::{side.get('id')}"


def mention_to_identity(identities: List[Dict[str, Any]]) -> Dict[str, str]:
    """{'entry::local_id' -> global identity id}.

    The review queue keys on MENTIONS. Grouping by mention is not "per person":
    a person appearing in eight entries yields eight screens. Pass this map to
    group_by_person so one resolved person really is one screen.
    """
    out: Dict[str, str] = {}
    for ident in identities or []:
        # person_index.json uses `person_id`; accept the other spellings too
        gid = (ident.get("person_id") or ident.get("global_id")
               or ident.get("id") or ident.get("identity"))
        if gid is None:
            # Refuse to build a map of Nones. Silently mapping every mention to
            # None collapses the whole corpus onto one screen and then discards
            # every candidate as a self-match — which is exactly what happened
            # the first time this ran.
            raise ValueError(
                "identity records carry no id field (expected 'person_id'); "
                f"got keys {sorted(ident)[:8]}")
        for m in ident.get("mentions") or []:
            out[f"{m.get('entry')}::{m.get('id')}"] = str(gid)
    return out


def group_by_person(review_queue: List[Dict[str, Any]],
                    min_score: float = 0.0,
                    identity_of: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Regroup a pairwise review queue into one screen per person.

    Returns a list of {person, candidates: [...], best_score, n_candidates},
    sorted so the people with the strongest candidate come first. Each pair
    appears under BOTH of its people: the reviewer may reach the decision from
    either side, and the resulting must/cannot constraint is symmetric anyway.
    """
    by_person: Dict[str, Dict[str, Any]] = {}
    for pair in review_queue:
        score = pair.get("score", 0.0)
        if score < min_score:
            continue
        for near, far in (("a", "b"), ("b", "a")):
            side, other = pair.get(near) or {}, pair.get(far) or {}
            if not side or not other:
                continue
            mk = _key(side)
            # one screen per RESOLVED person when the map is supplied, else per
            # mention (which over-counts anyone appearing in several entries)
            k = (identity_of or {}).get(mk, mk)
            slot = by_person.setdefault(k, {
                "person": {"entry": side.get("entry"), "id": side.get("id"),
                           "name": side.get("name"), "detail": side.get("detail"),
                           "identity": k if identity_of else None},
                "candidates": [], "_seen": set(),
            })
            ok = (identity_of or {}).get(_key(other), _key(other))
            if ok == k or ok in slot["_seen"]:
                continue      # same person, or this candidate already listed
            slot["_seen"].add(ok)
            slot["candidates"].append({
                "entry": other.get("entry"), "id": other.get("id"),
                "name": other.get("name"), "detail": other.get("detail"),
                "score": score, "reasons": pair.get("reasons") or [],
            })
    screens = []
    for slot in by_person.values():
        cands = sorted(slot["candidates"], key=lambda c: -c["score"])
        slot.pop("_seen", None)
        screens.append({**slot, "candidates": cands,
                        "n_candidates": len(cands),
                        "best_score": cands[0]["score"] if cands else 0.0})
    screens.sort(key=lambda s: (-s["best_score"], -s["n_candidates"]))
    return screens


def summarize(screens: List[Dict[str, Any]], total_identities: Optional[int] = None,
              total_pairs: Optional[int] = None) -> Dict[str, Any]:
    """Honest accounting of what a reviewer actually faces.

    `screens_with_candidates` is the real workload. `total_identities` is passed
    in rather than inferred, because identities with no candidate never appear
    in the review queue and so cannot be counted from it.
    """
    n = len(screens)
    dist = defaultdict(int)
    for s in screens:
        c = s["n_candidates"]
        dist["0" if c == 0 else "1" if c == 1 else "2-5" if c <= 5
             else "6-20" if c <= 20 else "21+"] += 1
    out = {
        "screens_with_candidates": n,
        "candidate_rows_total": sum(s["n_candidates"] for s in screens),
        "candidates_per_screen": round(sum(s["n_candidates"] for s in screens) / n, 1) if n else 0.0,
        "distribution": dict(dist),
    }
    if total_identities is not None:
        out["total_identities"] = total_identities
        out["identities_needing_no_review"] = max(0, total_identities - n)
    if total_pairs is not None:
        out["pair_queue_length"] = total_pairs
        # each pair is shown under both people, so rows = 2 x pairs; the saving
        # is in DECISIONS per screen, not in rows rendered.
        out["decisions_if_reviewed_per_pair"] = total_pairs
    return out


def format_summary(rep: Dict[str, Any]) -> str:
    lines = ["=" * 60, "Per-person review surface", "=" * 60]
    if "pair_queue_length" in rep:
        lines.append(f"pairwise queue:            {rep['pair_queue_length']:,} decisions")
    lines.append(f"people needing review:     {rep['screens_with_candidates']:,} screens")
    if "identities_needing_no_review" in rep:
        lines.append(f"people with no candidate:  {rep['identities_needing_no_review']:,} "
                     f"(of {rep['total_identities']:,} identities)")
    lines.append(f"candidates per screen:     {rep['candidates_per_screen']} average")
    d = rep.get("distribution", {})
    if d:
        lines.append("screens by candidate count: "
                     + ", ".join(f"{k}: {v:,}" for k, v in sorted(d.items())))
    lines.append("=" * 60)
    return "\n".join(lines)
