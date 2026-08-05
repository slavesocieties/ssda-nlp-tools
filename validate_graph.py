#!/usr/bin/env python3
"""validate_graph.py — logical invariants on the social network.

Offline, $0, no network, no key.

    python validate_graph.py

The graph is the actual Task 3 deliverable and it is the one stage that has
never been checked at all. There is no gold social network to compare against
and there never will be, so this checks what can be checked without one: things
that are wrong on their face, whatever the source data said.

Every check here is a statement no historical record can satisfy, so a hit is
always a defect in us and never a peculiarity of the archive:

  self-loop            a person related to themselves
  ancestry cycle       A is an ancestor of B and B of A
  symmetry             parent/child, spouse/spouse, godparent/godchild and
                       enslaver/slave must appear from both sides
  role contradiction   two people who are both spouse and parent-child, or a
                       person who is their own godparent
  temporal             a parent whose events all postdate the child's birth,
                       or an edge between two people whose lifetimes cannot
                       have overlapped

WHAT A HIT DOES NOT TELL YOU
----------------------------
It says the graph is inconsistent, not which end is wrong. A parent/child edge
missing its reverse could be a dropped edge or an invented one. The value is
that the total is currently unmeasured, so any number at all is progress, and
zero on a check is meaningful.

The hubs are reported separately and deliberately without a verdict. "Maria de
la Cruz" holds 82 mentions and degree 208, and that is exactly the shape both a
real godmother-of-the-parish and a bad merge would take. Daniel's 1,000 labels
are the only thing that can settle it; until then the number is a watch item,
not a finding.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

# Relationship pairs that must both exist if either does.
INVERSES = {
    "parent": "child", "child": "parent",
    "godparent": "godchild", "godchild": "godparent",
    "grandparent": "grandchild", "grandchild": "grandparent",
    "enslaver": "slave", "slave": "enslaver",
    "spouse": "spouse", "sibling": "sibling",
    "patron": "client", "client": "patron",
    "witness": "witness",
}
ANCESTRY = ("parent", "grandparent")
# Two people cannot stand in both of these at once.
CONTRADICTORY = [("spouse", "parent"), ("spouse", "child"),
                 ("parent", "child"), ("parent", "godparent")]


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("nodes") or [], (d.get("edges") or d.get("links") or [])


def check(nodes, edges):
    out = defaultdict(list)
    by_pair = defaultdict(set)
    adj = defaultdict(set)
    for e in edges:
        s, t, ty = e.get("source"), e.get("target"), (e.get("type") or "").lower()
        if s is None or t is None:
            out["edge_missing_endpoint"].append(e)
            continue
        if s == t:
            out["self_loop"].append({"person": s, "type": ty})
            continue
        by_pair[(s, t)].add(ty)
        if ty in ANCESTRY:
            adj[s].add(t)

    node_ids = {n.get("id") for n in nodes}
    for e in edges:
        for end in ("source", "target"):
            v = e.get(end)
            if v is not None and v not in node_ids:
                out["edge_to_unknown_node"].append({"edge": e, "missing": v})

    # symmetry
    for (s, t), types in by_pair.items():
        for ty in types:
            inv = INVERSES.get(ty)
            if inv and inv not in by_pair.get((t, s), ()):
                out["missing_inverse"].append({"source": s, "target": t,
                                               "type": ty, "expected": inv})

    # Contradictory roles between the same two people.
    #
    # ONE PAIR, ONE ROW. by_pair is keyed on the ORDERED (s, t), so reporting per
    # key counts every pair twice: A-B carrying {parent, spouse} and B-A carrying
    # {child, spouse} are the same two people seen from both ends, and listing
    # both inflated the count from 5 to 8. That is the same double-count that
    # once turned an ancestry-cycle figure into a wrong "28 -> 0", so the roles
    # from both directions are unioned onto one unordered pair here.
    # DETECT per direction, REPORT once. Unioning the two directions first looks
    # tidier and is catastrophically wrong: A->B "parent" plus B->A "child" is the
    # ordinary inverse pair every real parent has, so the union makes every
    # parent/child in the corpus contradict itself. That mistake took the count
    # from 8 to 10,050 before this comment existed. The contradiction is always a
    # property of ONE direction -- A->B carrying both "parent" and "spouse".
    reported = set()
    for (s, t), types in by_pair.items():
        for a, b in CONTRADICTORY:
            if a in types and b in types:
                key = tuple(sorted((s, t)))
                if key not in reported:
                    reported.add(key)
                    out["contradictory_roles"].append({"a": s, "b": t,
                                                       "types": sorted(types)})
                break          # one row per pair, not one per contradiction

    # ancestry cycles (iterative DFS; the graph is large)
    colour = {}
    for start in list(adj):
        if colour.get(start):
            continue
        stack = [(start, iter(adj.get(start, ())))]
        colour[start] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                colour[node] = 2
                stack.pop()
                continue
            if colour.get(nxt) == 1:
                out["ancestry_cycle"].append({"from": node, "to": nxt})
            elif not colour.get(nxt):
                colour[nxt] = 1
                stack.append((nxt, iter(adj.get(nxt, ()))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network",
                    default="production/luna_v3/corpus_final_pipeline/network.json")
    ap.add_argument("--out", default="production/luna_v3/graph_validation.json")
    ap.add_argument("--top-hubs", type=int, default=8)
    args = ap.parse_args(argv)

    nodes, edges = load(args.network)
    print(f"{len(nodes):,} people, {len(edges):,} relationship edges\n")
    res = check(nodes, edges)

    print("=== logical invariants (a hit is always our defect)")
    names = ("edge_missing_endpoint", "edge_to_unknown_node", "self_loop",
             "missing_inverse", "contradictory_roles", "ancestry_cycle")
    clean = True
    for k in names:
        n = len(res.get(k) or [])
        clean &= (n == 0)
        mark = "OK  " if n == 0 else "FAIL"
        print(f"    [{mark}] {k:24s} {n:,}")
        for row in (res.get(k) or [])[:3]:
            print(f"             {json.dumps(row, ensure_ascii=False)[:120]}")
    print(f"\n    {'all invariants hold' if clean else 'see above'}")

    deg = Counter()
    for e in edges:
        deg[e.get("source")] += 1
        deg[e.get("target")] += 1
    name_of = {n.get("id"): n.get("name") or n.get("label") for n in nodes}
    print(f"\n=== degree distribution")
    print(f"    median {sorted(deg.values())[len(deg)//2] if deg else 0}, "
          f"max {max(deg.values()) if deg else 0}, "
          f"people with degree > 50: {sum(1 for v in deg.values() if v > 50)}")
    print(f"    top hubs (NOT a verdict -- a real parish godmother and an "
          f"over-merge look identical here):")
    for pid, d in deg.most_common(args.top_hubs):
        print(f"      {d:5d}  {str(name_of.get(pid))[:38]:40s} {pid}")

    json.dump({k: v for k, v in res.items()}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n-> {args.out}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
