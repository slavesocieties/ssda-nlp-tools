"""Automated, confidence-scored cross-entry person disambiguation.

Replaces utility.py's manual ``input("y/n")`` merge hook with a scalable
pipeline: score every candidate mention pair, auto-merge the confident ones,
and route the borderline ones to a ranked review queue instead of blocking on a
human for all of them.

Pairwise score = name similarity, adjusted by attribute *compatibility* (hard
conflicts like free vs enslaved or different phenotype push apart; agreement
pulls together) and shared-relationship context (two people enslaved by the same
named enslaver are likely linkable). Mentions from the *same* entry are never
merged — the extractor already separated them.

Design choices called out honestly:
  * Blocking by first name-token keeps this near-linear; a phonetic key
    (double-metaphone) is the natural upgrade for spelling drift across blocks.
  * Auto-merge uses union-find, so links are transitive. We record each
    cluster's weakest internal edge and flag clusters whose cohesion dips below
    `auto_threshold` for review, rather than trusting the chain blindly.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .textmatch import name_similarity, name_tokens, normalize_name, phonetic_key

# Attributes that, when both present and different, are evidence of DIFFERENT people.
HARD_ATTRS = ["phenotype", "free", "ethnicity", "origin", "occupation", "legitimate", "rank"]


class _UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def _val(p, k):
    v = p.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().lower()


# Relationship types whose counterpart identifies a person strongly: an enslaved
# "Juan" is distinguished by WHO enslaves him; a spouse by whom they married.
DISCRIMINATIVE_CTX = ("enslaver", "spouse", "parent")


def _ctx_by_type(ctx):
    d: Dict[str, set] = defaultdict(set)
    for t, n in ctx or ():
        if t in DISCRIMINATIVE_CTX:
            d[t].add(n)
    return d


def _third_party_same(x: str, y: str) -> bool:
    """Are two third-party names (a spouse/enslaver/parent) the same person?

    Estate surnames are shared by everyone attached to the estate ("hanna
    macqueen" vs "rachael macqueen" are DIFFERENT wives), so the GIVEN name must
    agree; but short forms contain long forms ("rachael" is "rachael macqueen").
    """
    from difflib import SequenceMatcher
    from .textmatch import phonetic_fold
    tx, ty = name_tokens(x), name_tokens(y)
    if not tx or not ty:
        return False
    if set(tx) <= set(ty) or set(ty) <= set(tx):     # containment / short form
        return True
    given = SequenceMatcher(None, tx[0], ty[0]).ratio()
    if given >= 0.75 or phonetic_fold(tx[0]) == phonetic_fold(ty[0]):
        return name_similarity(x, y) >= 0.6
    return False


def _namesets_overlap(sa: set, sb: set) -> bool:
    for x in sa:
        for y in sb:
            if _third_party_same(x, y):
                return True
    return False


def pair_score(a: dict, b: dict, a_rel_ctx=None, b_rel_ctx=None) -> Tuple[float, List[str]]:
    """Return (score in [0,1], reasons). Higher = more likely the same person."""
    reasons = []
    nsim = name_similarity(a.get("name"), b.get("name"))
    if nsim <= 0.0:
        return 0.0, ["different names"]

    score = nsim
    reasons.append(f"name~{nsim:.2f}")

    # Attribute agreement weighted by informativeness: phenotype/free agree for
    # nearly everyone in an enslaved-population register, so they barely count;
    # occupation/rank are rarer and mean more.
    AGREE_W = {"phenotype": 0.02, "free": 0.02, "legitimate": 0.02,
               "origin": 0.04, "ethnicity": 0.04, "occupation": 0.08, "rank": 0.08}
    agree = conflict = 0
    for k in HARD_ATTRS:
        va, vb = _val(a, k), _val(b, k)
        if va is not None and vb is not None:
            if va == vb:
                agree += 1
                score += AGREE_W.get(k, 0.03)
            else:
                conflict += 1
                reasons.append(f"conflict:{k}({va}!={vb})")
    score -= 0.25 * conflict

    # shared relationship context: same-typed edge to a same-named third party
    ctx_overlap = False
    if a_rel_ctx and b_rel_ctx:
        shared = a_rel_ctx & b_rel_ctx
        if shared:
            ctx_overlap = True
            score += min(0.15, 0.05 * len(shared))
            reasons.append(f"shared_rel({len(shared)})")
        # typed comparison: same rel type on both sides but pointing at
        # non-overlapping third parties = near-definitive evidence of DIFFERENT
        # people in this domain (your enslaver / spouse / parents don't vary by
        # entry), so it must outweigh any pile of weak attribute agreements.
        ca, cb = _ctx_by_type(a_rel_ctx), _ctx_by_type(b_rel_ctx)
        for t in set(ca) & set(cb):
            if _namesets_overlap(ca[t], cb[t]):
                ctx_overlap = True
            else:
                score -= 0.35
                reasons.append(f"ctx-conflict:{t}({'/'.join(sorted(ca[t])[:1])}"
                               f"!={'/'.join(sorted(cb[t])[:1])})")

    # single bare-name guard: "Juan" ~ "Juan" carries little identity on its own,
    # and shared phenotype/free/origin is population-universal here, not personal.
    # Require CONTEXT corroboration (same spouse/enslaver/parents) to auto-merge;
    # otherwise cap below auto threshold so the pair lands in review. This also
    # stops context-empty mentions from acting as transitive union-find bridges.
    single = len(name_tokens(a.get("name"))) == 1 and len(name_tokens(b.get("name"))) == 1
    if single and not ctx_overlap:
        if score > 0.82:
            score = 0.82
            reasons.append("capped: single-token name, no context corroboration")

    return max(0.0, min(1.0, score)), reasons


def _mentions_from_volume(volume: dict) -> List[dict]:
    """Flatten entries -> mentions, attaching a relationship context set per mention."""
    entries = volume.get("entries") or volume.get("examples") or []
    mentions = []
    for e in entries:
        # examples-format rows carry the per-entry id in "entry" ("0013-01") and the
        # *volume* id in "id" (239746); volume-record rows carry the entry id in "id".
        # Prefer "entry" so we don't collapse a whole volume into one pseudo-entry.
        eid = str(e.get("entry") or e.get("id") or "")
        data = e.get("data") or {}
        people = data.get("people", []) or []
        events = data.get("events", []) or []
        id_to_name = {str(p.get("id")): normalize_name(p.get("name")) for p in people}
        # person ids that are principals of a once-in-a-lifetime sacrament: you are
        # baptized/born/buried exactly once, so two such mentions in DIFFERENT
        # entries cannot be the same person (a strong precision constraint).
        unique_sacrament_pids = set()
        for ev in events:
            if str(ev.get("type", "")).lower() in ("baptism", "birth", "burial"):
                for pid in ev.get("principals", []) or []:
                    unique_sacrament_pids.add(str(pid))
        for p in people:
            # context = set of (rel_type, related-person-normalized-name)
            ctx = set()
            for r in p.get("relationships", []) or []:
                if isinstance(r, dict):
                    rn = id_to_name.get(str(r.get("related_person")))
                    rt = r.get("relationship_type")
                    if rn and rt:
                        ctx.add((str(rt).lower(), rn))
            m = dict(p)
            m["_entry"] = eid
            m["_local_id"] = str(p.get("id"))
            m["_ctx"] = ctx
            m["_unique_sacrament"] = str(p.get("id")) in unique_sacrament_pids
            mentions.append(m)
    return mentions


def _merge_attributes(members: List[dict]) -> Dict[str, Any]:
    """Merge attribute values across a cluster, recording conflicts as lists."""
    merged: Dict[str, Any] = {}
    conflicts: Dict[str, list] = {}
    for k in HARD_ATTRS + ["age"]:
        vals = []
        for m in members:
            v = m.get(k)
            if v is not None and not (isinstance(v, str) and not v.strip()):
                vals.append(v)
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) == 1:
            merged[k] = uniq[0]
        elif len(uniq) > 1:
            merged[k] = uniq          # keep all; downstream/human resolves
            conflicts[k] = uniq
    # titles = union
    titles = []
    for m in members:
        for t in (m.get("titles") or []):
            if t not in titles:
                titles.append(t)
    if titles:
        merged["titles"] = titles
    return merged, conflicts


def _snapshot(m: dict) -> dict:
    """Compact evidence card for a mention, for human review."""
    snap = {k: m.get(k) for k in ("occupation", "phenotype", "free", "origin",
                                  "ethnicity", "age", "legitimate")
            if m.get(k) is not None}
    if m.get("titles"):
        snap["titles"] = m["titles"]
    ctx = sorted(m.get("_ctx") or ())
    if ctx:
        snap["context"] = [f"{t}: {n}" for t, n in ctx]
    return snap


def disambiguate_volume(
    volume: Any,
    auto_threshold: float = 0.86,
    review_threshold: float = 0.70,
    volume_tag: Optional[str] = None,
    constraints: Optional[dict] = None,
) -> Dict[str, Any]:
    """Cluster person mentions into identities.

    Returns {identities, review_queue, stats}. Pairs at or above auto_threshold
    are merged; pairs in [review_threshold, auto_threshold) become review items.

    `constraints` carries human review decisions back in:
        {"must":   [[{"entry","id"}, {"entry","id"}], ...],   # same person
         "cannot": [[{"entry","id"}, {"entry","id"}], ...]}   # different people
    Must-links are unioned outright; cannot-links are excluded from auto-merge
    and any cluster that still joins them transitively is flagged for review.
    """
    if isinstance(volume, str):
        with open(volume, "r", encoding="utf-8") as f:
            volume = json.load(f)
    tag = volume_tag or str(volume.get("id", "V"))

    mentions = _mentions_from_volume(volume)
    n = len(mentions)
    uf = _UnionFind(n)

    key_to_idx = {(m["_entry"], m["_local_id"]): i for i, m in enumerate(mentions)}

    def _pair_idx(pair):
        a = key_to_idx.get((str(pair[0]["entry"]), str(pair[0]["id"])))
        b = key_to_idx.get((str(pair[1]["entry"]), str(pair[1]["id"])))
        return (a, b) if a is not None and b is not None else None

    must_pairs = [p for p in ((constraints or {}).get("must") or []) if _pair_idx(p)]
    cannot_set = set()
    for p in (constraints or {}).get("cannot") or []:
        idx = _pair_idx(p)
        if idx:
            cannot_set.add(frozenset(idx))

    # block by PHONETIC key of the first name-token to avoid O(n^2) while still
    # grouping scribal variants (Gonzalez/Gonzales) into the same candidate block
    blocks: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(mentions):
        blocks[phonetic_key(m.get("name"))].append(i)

    review_queue = []
    auto_edges: List[Tuple[int, int, float]] = []
    for _, idxs in blocks.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if mentions[i]["_entry"] == mentions[j]["_entry"]:
                    continue  # never merge two people from the same entry
                s, reasons = pair_score(mentions[i], mentions[j],
                                        mentions[i]["_ctx"], mentions[j]["_ctx"])
                # once-in-a-lifetime sacrament guard: two baptism/birth/burial
                # principals from different entries are different people (you are
                # baptized once). Block auto-merge; keep very-similar pairs visible
                # in the review queue (could be a double-recorded entry).
                if (mentions[i].get("_unique_sacrament") and mentions[j].get("_unique_sacrament")
                        and s >= review_threshold):
                    review_queue.append({
                        "score": round(min(s, auto_threshold - 0.01), 3),
                        "reasons": reasons + ["blocked: both sacrament principals"],
                        "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                              "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                        "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                              "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                    })
                    continue
                if s >= auto_threshold:
                    if frozenset((i, j)) in cannot_set:
                        continue          # human already ruled: different people
                    uf.union(i, j)
                    auto_edges.append((i, j, s))
                elif s >= review_threshold:
                    review_queue.append({
                        "score": round(s, 3),
                        "reasons": reasons,
                        "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                              "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                        "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                              "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                    })

    # human decisions: must-links union outright and settle their review items
    for p in must_pairs:
        a, b = _pair_idx(p)
        uf.union(a, b)
        auto_edges.append((a, b, 1.0))
    decided = {frozenset(_pair_idx(p)) for p in must_pairs} | set(cannot_set)
    if decided:
        review_queue = [r for r in review_queue
                        if frozenset((key_to_idx.get((r["a"]["entry"], r["a"]["id"])),
                                      key_to_idx.get((r["b"]["entry"], r["b"]["id"])))) not in decided]

    # gather clusters
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # a cannot-link that still ended up in one cluster (via a transitive chain)
    # is a conflict a human must untangle — flag the cluster, don't hide it
    violated_roots = set()
    for pair in cannot_set:
        a, b = tuple(pair)
        if uf.find(a) == uf.find(b):
            violated_roots.add(uf.find(a))

    # cluster cohesion: weakest internal auto edge (for flagging chained merges)
    cohesion: Dict[int, float] = {}
    for i, j, s in auto_edges:
        root = uf.find(i)
        cohesion[root] = min(cohesion.get(root, 1.0), s)

    identities = []
    flagged = 0
    for k, (root, idxs) in enumerate(sorted(clusters.items()), 1):
        members = [mentions[i] for i in idxs]
        merged_attrs, conflicts = _merge_attributes(members)
        # canonical name = the longest (most complete) surface form
        canonical = max((m.get("name") or "" for m in members), key=len)
        coh = cohesion.get(root, 1.0) if len(idxs) > 1 else 1.0
        needs_review = (len(idxs) > 1 and coh < auto_threshold) or root in violated_roots
        if needs_review:
            flagged += 1
        identities.append({
            "person_id": f"{tag}-{k:04d}",
            "canonical_name": canonical,
            "n_mentions": len(idxs),
            "mentions": [{"entry": m["_entry"], "id": m["_local_id"], "name": m.get("name")}
                         for m in members],
            "attributes": merged_attrs,
            "attribute_conflicts": conflicts,
            "cluster_cohesion": round(coh, 3),
            "needs_review": needs_review,
        })

    multi = [i for i in identities if i["n_mentions"] > 1]
    return {
        "identities": identities,
        "review_queue": sorted(review_queue, key=lambda x: -x["score"]),
        "stats": {
            "mentions": n,
            "identities": len(identities),
            "merged_identities": len(multi),
            "auto_merges": len(auto_edges),
            "review_pairs": len(review_queue),
            "flagged_clusters": flagged,
            "reduction": round(1 - len(identities) / n, 4) if n else 0.0,
        },
    }


def format_disambiguation(result: Dict[str, Any], top: int = 12) -> str:
    s = result["stats"]
    lines = ["=" * 60, "Person disambiguation", "=" * 60,
             f"mentions:            {s['mentions']}",
             f"distinct identities: {s['identities']}  "
             f"(mention->identity reduction {s['reduction']*100:.1f}%)",
             f"merged identities:   {s['merged_identities']}  "
             f"(from {s['auto_merges']} auto-merge links)",
             f"review queue:        {s['review_pairs']} borderline pairs",
             f"flagged clusters:    {s['flagged_clusters']} (weak internal link)",
             ""]
    multi = [i for i in result["identities"] if i["n_mentions"] > 1]
    if multi:
        lines.append("merged identities (top by mention count):")
        for idn in sorted(multi, key=lambda x: -x["n_mentions"])[:top]:
            flag = "  ⚑REVIEW" if idn["needs_review"] else ""
            where = ", ".join(f"{m['entry']}:{m['id']}" for m in idn["mentions"])
            lines.append(f"  {idn['person_id']}  {idn['canonical_name']!r}  "
                         f"x{idn['n_mentions']} [{where}]{flag}")
            if idn["attribute_conflicts"]:
                lines.append(f"        conflicts: {idn['attribute_conflicts']}")
    if result["review_queue"]:
        lines.append("")
        lines.append(f"review queue (top {min(top, len(result['review_queue']))} by score):")
        for r in result["review_queue"][:top]:
            lines.append(f"  {r['score']:.2f}  {r['a']['name']!r}({r['a']['entry']}) ~ "
                         f"{r['b']['name']!r}({r['b']['entry']})  {r['reasons']}")
    lines.append("=" * 60)
    return "\n".join(lines)
