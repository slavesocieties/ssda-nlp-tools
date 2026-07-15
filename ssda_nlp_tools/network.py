"""Build the cross-entry person/relationship social graph from a resolved volume.

This is the payoff of the whole stage: once mentions are resolved to volume-wide
identities, the per-entry relationships collapse into ONE graph where a person is
a node and every stated relationship is a typed edge. The officiating priest
becomes a hub linking dozens of baptisms; an enslaver links to every enslaved
person named across the book. Exports GraphML (stdlib xml — loads in Gephi,
networkx, Cytoscape) plus a summary JSON. No third-party deps.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .resolve import resolve_volume
from .textmatch import normalize_name


class _UF:
    def __init__(self): self.p = {}
    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_network(resolved: Any, **kwargs) -> Dict[str, Any]:
    """Build a graph from a resolved volume (or resolve one on the fly).

    `resolved` may be the dict returned by resolve_volume, or a raw volume /
    path (in which case we resolve it first, passing kwargs through).
    """
    if isinstance(resolved, str) or "person_index" not in (resolved or {}):
        resolved = resolve_volume(resolved, **kwargs)

    index = {i["person_id"]: i for i in resolved["person_index"]}
    volume = resolved["volume"]
    entries = volume.get("entries") or volume.get("examples") or []

    # nodes -------------------------------------------------------------------
    nodes: Dict[str, dict] = {}
    for pid, ident in index.items():
        attrs = ident.get("attributes", {})
        nodes[pid] = {
            "id": pid,
            "label": ident.get("canonical_name", ""),
            "mentions": ident.get("n_mentions", 1),
            "needs_review": ident.get("needs_review", False),
            # flatten list-valued (conflicting) attrs for a scalar graph attribute
            **{k: (", ".join(map(str, v)) if isinstance(v, list) else v)
               for k, v in attrs.items() if k != "titles"},
        }

    # edges: (subj_gid, type, obj_gid) -> {weight, entries} -------------------
    edges: Dict[Tuple[str, str, str], dict] = {}
    dropped_unresolved = 0
    for e in entries:
        eid = str(e.get("entry") or e.get("id") or "")
        people = (e.get("data") or {}).get("people", []) or []
        local_to_global = {str(p.get("id")): p.get("global_id") for p in people}
        for p in people:
            subj = p.get("global_id")
            for r in p.get("relationships", []) or []:
                if not isinstance(r, dict):
                    continue
                obj = local_to_global.get(str(r.get("related_person")))
                rtype = r.get("relationship_type")
                if not (subj and obj and rtype) or subj == obj:
                    dropped_unresolved += 1 if not (subj and obj) else 0
                    continue
                key = (subj, str(rtype), obj)
                slot = edges.setdefault(key, {"weight": 0, "entries": set()})
                slot["weight"] += 1
                slot["entries"].add(eid)

    edge_list = [{"source": s, "type": t, "target": o,
                  "weight": v["weight"], "entries": sorted(v["entries"])}
                 for (s, t, o), v in edges.items()]

    # graph stats -------------------------------------------------------------
    deg = Counter()
    uf = _UF()
    for ed in edge_list:
        deg[ed["source"]] += 1
        deg[ed["target"]] += 1
        uf.union(ed["source"], ed["target"])
    comps = defaultdict(list)
    for pid in nodes:
        comps[uf.find(pid)].append(pid)
    comp_sizes = sorted((len(v) for v in comps.values()), reverse=True)

    hubs = [{"id": pid, "label": nodes[pid]["label"], "degree": d,
             "mentions": nodes[pid]["mentions"]}
            for pid, d in deg.most_common(10)]
    rel_types = Counter(ed["type"] for ed in edge_list)
    cross_entry_people = sum(1 for n in nodes.values() if n["mentions"] > 1)
    cross_entry_edges = sum(1 for ed in edge_list if len(ed["entries"]) > 1)

    return {
        "nodes": list(nodes.values()),
        "edges": edge_list,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edge_list),
            "relationship_instances": sum(ed["weight"] for ed in edge_list),
            "isolated_nodes": sum(1 for pid in nodes if deg[pid] == 0),
            "connected_components": len(comp_sizes),
            "largest_component": comp_sizes[0] if comp_sizes else 0,
            "cross_entry_people": cross_entry_people,
            "cross_entry_edges": cross_entry_edges,
            "relationship_types": dict(rel_types),
            "top_hubs": hubs,
            "dropped_unresolved_edges": dropped_unresolved,
        },
    }


# --------------------------------------------------------------------------- #
# GraphML export (stdlib xml)
# --------------------------------------------------------------------------- #

_NS = "http://graphml.graphdrawing.org/xmlns"


def to_graphml(network: Dict[str, Any], path: str) -> str:
    node_attrs = ["label", "mentions", "needs_review", "occupation", "phenotype",
                  "free", "origin", "ethnicity", "age", "legitimate", "rank"]
    ET.register_namespace("", _NS)
    root = ET.Element(f"{{{_NS}}}graphml")

    def key(kid, name, target, typ):
        k = ET.SubElement(root, f"{{{_NS}}}key")
        k.set("id", kid); k.set("for", target)
        k.set("attr.name", name); k.set("attr.type", typ)

    for a in node_attrs:
        key(a, a, "node", "string")
    key("etype", "relationship_type", "edge", "string")
    key("weight", "weight", "edge", "double")
    key("entries", "entries", "edge", "string")

    g = ET.SubElement(root, f"{{{_NS}}}graph")
    g.set("edgedefault", "directed")

    for n in network["nodes"]:
        ge = ET.SubElement(g, f"{{{_NS}}}node"); ge.set("id", n["id"])
        for a in node_attrs:
            if n.get(a) is not None:
                d = ET.SubElement(ge, f"{{{_NS}}}data"); d.set("key", a)
                d.text = str(n[a])
    for i, ed in enumerate(network["edges"]):
        ee = ET.SubElement(g, f"{{{_NS}}}edge")
        ee.set("id", f"e{i}"); ee.set("source", ed["source"]); ee.set("target", ed["target"])
        for k, v in (("etype", ed["type"]), ("weight", ed["weight"]),
                     ("entries", ";".join(ed["entries"]))):
            d = ET.SubElement(ee, f"{{{_NS}}}data"); d.set("key", k); d.text = str(v)

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def format_network(network: Dict[str, Any], top: int = 10) -> str:
    s = network["stats"]
    lines = ["=" * 60, "Person / relationship network", "=" * 60,
             f"nodes (people):        {s['nodes']}",
             f"edges (relationships): {s['edges']} unique typed "
             f"({s['relationship_instances']} instances)",
             f"isolated nodes:        {s['isolated_nodes']}",
             f"connected components:  {s['connected_components']} "
             f"(largest {s['largest_component']} people)",
             f"cross-entry people:    {s['cross_entry_people']} "
             f"(appear in >1 entry — the links that matter)",
             f"cross-entry edges:     {s['cross_entry_edges']}",
             ""]
    if s["relationship_types"]:
        rt = ", ".join(f"{k}={v}" for k, v in sorted(s["relationship_types"].items(),
                                                     key=lambda x: -x[1]))
        lines.append(f"relationship types: {rt}")
        lines.append("")
    lines.append(f"top hubs (by degree):")
    for h in s["top_hubs"][:top]:
        lines.append(f"  {h['degree']:>3}  {h['label']!r}  ({h['id']}, x{h['mentions']} mentions)")
    lines.append("=" * 60)
    return "\n".join(lines)
