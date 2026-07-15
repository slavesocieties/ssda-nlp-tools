"""Corrected, dependency-free versions of the buggy date + relationship helpers.

These mirror the signatures in the existing utility.py / extract.py so they can
be dropped in later, but are pure (no file I/O, no logging side effects) so they
are unit-testable. Each returns its result plus a list of change descriptions.

Bugs fixed vs the originals (all confirmed by running the originals):
  * parse_date: returned STRINGS, not ints (the `for part in parts: part=int(part)`
    loop rebinds the loop variable and mutates nothing); and crashed on any date
    RANGE because it appended `int(parts)` (the whole list) instead of `int(part)`.
  * complete_date: therefore raised TypeError on any string date with a month
    (`date[1] - 1` on a str).
  * fix_relationships: `is_principal` consulted only `events[0]`, so entries with
    multiple events (e.g. baptism + birth, or a marriage) mis-repaired reciprocity.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

RECIPROCAL_RELS = {
    "parent": "child", "child": "parent",
    "grandparent": "grandchild", "grandchild": "grandparent",
    "enslaver": "slave", "slave": "enslaver",
    "indenturer": "indentured servant", "indentured servant": "indenturer",
    "spouse": "spouse",
    "godparent": "godchild", "godchild": "godparent",
}
_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def parse_date(date: str) -> List[int]:
    """ISO-8601 date or 'start/end' range -> list of ints (correct)."""
    def _ints(part_str: str) -> List[int]:
        return [int(x) for x in part_str.split("-") if x != ""]

    if "/" in date:
        start, end = date.split("/", 1)
        return _ints(start) + _ints(end)
    return _ints(date)


def compare_dates(x: List[int], y: List[int]) -> bool:
    """True if x is on or before y (component-wise year, month, day)."""
    for a, b in zip(x, y):
        if a < b:
            return True
        if a > b:
            return False
    return True  # equal on shared components


def complete_date(date, mode: str = "m"):
    """Complete an incomplete date without inference. mode: 's' start, 'e' end, 'm' range."""
    if isinstance(date, str):
        date = parse_date(date)
    date = [int(x) for x in date]

    if mode == "s":
        return (date[0], 1, 1) if len(date) == 1 else (date[0], date[1], 1)
    if mode == "e":
        if len(date) == 1:
            return (date[0], 12, 31)
        return (date[0], date[1], _MONTH_DAYS[date[1] - 1])
    # mode == "m": produce a [start, end] range covering the missing precision
    if len(date) == 1:
        return (date[0], 1, 1, date[0], 12, 31)
    if len(date) == 2:
        return (date[0], date[1], 1, date[0], date[1], _MONTH_DAYS[date[1] - 1])
    return (date[0], date[1], date[2], date[0], date[1], date[2])


def is_principal(person_id: str, events: List[dict]) -> bool:
    """True if the person is a principal of ANY event (not just the first)."""
    for e in events or []:
        if person_id in (e.get("principals") or []):
            return True
    return False


def fix_relationships(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Reciprocity repair over ALL events. Pure; returns (fixed_data, changes).

    Rules (unchanged from the original intent):
      * A principal is the child/slave/godchild/spouse side of a relationship.
      * A one-directional relationship is assumed a miss, not a hallucination, so
        the reciprocal is added rather than the edge dropped.
      * Conflicting types between two non-principals are unfixable -> drop both.
    """
    data = copy.deepcopy(data)
    changes: List[str] = []
    people = data.get("people", [])
    events = data.get("events", []) or []

    # index current relationships as {pid: {other_pid: type}}
    rel = {}
    for p in people:
        rel[str(p.get("id"))] = {}
        for r in p.get("relationships", []) or []:
            if isinstance(r, dict) and r.get("related_person") and r.get("relationship_type"):
                rel[str(p["id"])][str(r["related_person"])] = r["relationship_type"]

    def _person(pid):
        for p in people:
            if str(p.get("id")) == pid:
                return p
        return None

    def del_relation(a, b):
        p = _person(a)
        if p and isinstance(p.get("relationships"), list):
            p["relationships"][:] = [r for r in p["relationships"]
                                     if str(r.get("related_person")) != b]

    def add_relation(a, b, rtype):
        del_relation(a, b)
        p = _person(a)
        if p is not None:
            p.setdefault("relationships", []).append(
                {"related_person": b, "relationship_type": rtype})

    # iterate over a snapshot; mutate people in place
    for p1 in list(rel.keys()):
        for p2, rtype in list(rel[p1].items()):
            if rtype not in RECIPROCAL_RELS:
                continue
            back = rel.get(p2, {}).get(p1)
            if back is None:
                changes.append(f"added reciprocal {RECIPROCAL_RELS[rtype]} for {p2}->{p1}")
                add_relation(p2, p1, RECIPROCAL_RELS[rtype])
                rel.setdefault(p2, {})[p1] = RECIPROCAL_RELS[rtype]
            elif back == RECIPROCAL_RELS[rtype]:
                continue  # already valid
            elif back != rtype or (not is_principal(p1, events) and not is_principal(p2, events)):
                changes.append(f"dropped unfixable {p1}<->{p2} ({rtype}/{back})")
                del_relation(p1, p2); del_relation(p2, p1)
                rel[p1].pop(p2, None); rel.get(p2, {}).pop(p1, None)
            else:
                principal = p1 if is_principal(p1, events) else p2
                other = p2 if principal == p1 else p1
                changes.append(f"fixed reciprocity {principal}/{other} ({rtype})")
                if rtype in ("child", "slave", "godchild"):
                    add_relation(principal, other, RECIPROCAL_RELS[rtype])
                    add_relation(other, principal, rtype)
                else:
                    add_relation(other, principal, RECIPROCAL_RELS[rtype])
                    add_relation(principal, other, rtype)

    return data, changes
