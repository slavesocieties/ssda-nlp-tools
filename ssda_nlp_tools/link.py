"""Cross-chunk / cross-volume person linking.

Takes several extraction outputs (page-range chunks of one volume today; whole
volumes tomorrow) and unifies them into ONE identity space and ONE social graph.

Strategy: normalize every input into a single combined volume whose entry ids
are collision-safe and carry chunk provenance, then run the already-verified
disambiguate -> resolve -> network chain on the combined volume. Cross-chunk
identity emerges from exactly the same scored evidence as within-chunk identity
(phonetic blocking, attribute conflicts, relationship context, the
once-in-a-lifetime sacrament guard) instead of a second bespoke matcher.

The registry it returns is the deliverable for cross-volume work: one row per
person with every (chunk, entry, local_id) mention behind it.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .network import build_network, format_network
from .resolve import resolve_volume


def _load(source: Any) -> dict:
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    return source


def _entries_of(vol: dict) -> List[dict]:
    return vol.get("entries") or vol.get("examples") or []


def _entry_id(e: dict) -> str:
    return str(e.get("entry") or e.get("id") or "")


def combine_volumes(sources: Sequence[Any], tags: Optional[Sequence[str]] = None) -> dict:
    """Merge chunk/volume JSONs into one combined volume.

    Entry ids are prefixed with their chunk tag only when needed to stay unique,
    but every combined entry always records its provenance in "chunk".
    """
    vols = [_load(s) for s in sources]
    if tags is None:
        tags = []
        for i, s in enumerate(sources):
            if isinstance(s, str):
                base = os.path.basename(s)
                tags.append(os.path.splitext(base)[0].replace("Generated_", "")[:24] or f"c{i}")
            else:
                tags.append(f"c{i}")

    seen_ids = set()
    combined_entries = []
    for vol, tag in zip(vols, tags):
        for e in _entries_of(vol):
            e = dict(e)                      # shallow copy; data is shared read-only
            eid = _entry_id(e)
            uid = eid if eid and eid not in seen_ids else f"{tag}:{eid or len(seen_ids)}"
            seen_ids.add(uid)
            e["entry"] = uid                 # normalized id key used downstream
            e.pop("id", None)                # avoid the volume-id/entry-id ambiguity
            e["chunk"] = tag
            combined_entries.append(e)

    base = vols[0] if vols else {}
    return {
        "type": base.get("type", "combined"),
        "id": base.get("id", "combined"),
        "title": f"combined[{', '.join(tags)}]",
        "chunks": list(tags),
        "entries": combined_entries,
    }


def link_volumes(
    sources: Sequence[Any],
    tags: Optional[Sequence[str]] = None,
    volume_tag: str = "LINKED",
    auto_threshold: float = 0.86,
    review_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Full cross-chunk link: returns {combined, resolved, network, registry, stats}."""
    combined = combine_volumes(sources, tags)
    resolved = resolve_volume(combined, volume_tag=volume_tag,
                              auto_threshold=auto_threshold,
                              review_threshold=review_threshold)
    net = build_network(resolved)

    # chunk provenance per entry id
    entry_chunk = {_entry_id(e): e.get("chunk", "?") for e in combined["entries"]}

    registry = []
    for ident in resolved["person_index"]:
        chunks = sorted({entry_chunk.get(m["entry"], "?") for m in ident["mentions"]})
        registry.append({
            "person_id": ident["person_id"],
            "canonical_name": ident["canonical_name"],
            "n_mentions": ident["n_mentions"],
            "chunks": chunks,
            "cross_chunk": len(chunks) > 1,
            "needs_review": ident["needs_review"],
            "attributes": ident["attributes"],
            "mentions": [{**m, "chunk": entry_chunk.get(m["entry"], "?")}
                         for m in ident["mentions"]],
        })

    cross = [r for r in registry if r["cross_chunk"]]
    stats = {
        **resolved["stats"],
        "chunks": combined["chunks"],
        "entries_combined": len(combined["entries"]),
        "cross_chunk_identities": len(cross),
    }
    return {"combined": combined, "resolved": resolved, "network": net,
            "registry": registry, "review_queue": resolved["review_queue"],
            "stats": stats}


def format_link(result: Dict[str, Any], top: int = 12) -> str:
    s = result["stats"]
    lines = ["=" * 64, "Cross-chunk person linking", "=" * 64,
             f"chunks:   {', '.join(s['chunks'])}",
             f"entries:  {s['entries_combined']}   mentions: {s['mentions']}",
             f"identities: {s['identities']}  (reduction {s['reduction']*100:.1f}%)",
             f"CROSS-CHUNK identities: {s['cross_chunk_identities']}  <- people linked across files",
             f"review queue: {s['review_pairs']} pairs", ""]
    cross = sorted([r for r in result["registry"] if r["cross_chunk"]],
                   key=lambda r: -r["n_mentions"])
    lines.append("people spanning chunks (top by mentions):")
    for r in cross[:top]:
        lines.append(f"  {r['person_id']}  {r['canonical_name']!r}  x{r['n_mentions']} "
                     f"across {r['chunks']}" + ("  ⚑REVIEW" if r["needs_review"] else ""))
    lines.append("")
    lines.append(format_network(result["network"], top=min(top, 8)))
    return "\n".join(lines)
