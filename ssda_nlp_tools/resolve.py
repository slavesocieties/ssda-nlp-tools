"""Apply disambiguation results back onto a volume.

Turns a volume of entry-siloed people (each entry has its own P01…) into a
*resolved* volume where every person mention also carries a stable, volume-wide
``global_id``, plus a compact ``person_index``. This is the join key that lets
the same enslaver / priest / godparent be followed across entries — the input
the social-network build (network.py) consumes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .disambiguate import disambiguate_volume


def _load(volume: Any) -> dict:
    if isinstance(volume, str):
        with open(volume, "r", encoding="utf-8") as f:
            return json.load(f)
    return volume


def resolve_volume(volume: Any, disamb: Optional[dict] = None, **kwargs) -> Dict[str, Any]:
    """Return {volume, person_index, stats}.

    `volume` is annotated in place-ish (a copy): each person gets ``global_id``.
    If `disamb` (output of disambiguate_volume) is not supplied it is computed.
    Extra kwargs pass through to disambiguate_volume (auto_threshold, etc.).
    """
    volume = _load(volume)
    if disamb is None:
        disamb = disambiguate_volume(volume, **kwargs)

    # (entry_id, local_id) -> global person_id
    lookup: Dict[tuple, str] = {}
    for ident in disamb["identities"]:
        for m in ident["mentions"]:
            lookup[(str(m["entry"]), str(m["id"]))] = ident["person_id"]

    entries = volume.get("entries") or volume.get("examples") or []
    n_annotated = 0
    for e in entries:
        eid = str(e.get("entry") or e.get("id") or "")
        for p in (e.get("data") or {}).get("people", []) or []:
            gid = lookup.get((eid, str(p.get("id"))))
            if gid:
                p["global_id"] = gid
                n_annotated += 1

    return {
        "volume": volume,
        "person_index": disamb["identities"],
        "review_queue": disamb["review_queue"],
        "stats": {**disamb["stats"], "annotated_mentions": n_annotated},
    }
