"""Show, in full, the merges the conflicting-relationship term prevented.

The A/B says the term adds 151 identities. That number is worthless to a
historian without the evidence behind it, and it is the evidence -- not the
count -- that tells us whether Daniel's ruling is implemented at the right
strength. So this prints both sides of every split with all the context the
scorer saw, plus the scored terms, so each one can be overruled by inspection.

Deliberately prints the WHOLE population when it is small enough rather than a
sample, because a sample of splits chosen by me is a sample chosen by the thing
under test.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, explain, score
from ssda_nlp_tools.volume_geo import load as load_geo


def clusters(path):
    d = json.load(open(path, encoding="utf-8"))
    out, seen = [], collections.Counter()
    for ident in (d["identities"] if isinstance(d, dict) else d):
        ms = []
        for m in ident["mentions"]:
            k = (m["entry"], str(m["id"]))
            seen[k] += 1
            ms.append(k + (seen[k],))
        out.append(frozenset(ms))
    return out


def describe(m):
    ctx = ", ".join(f"{r}={n}" for r, n in sorted(m.get("_ctx") or ()))
    attrs = " ".join(f"{k}={m.get(k)}" for k in
                     ("phenotype", "free", "ethnicity", "origin", "occupation")
                     if m.get(k) not in (None, "", "None"))
    return (f"      {str(m.get('name'))[:34]:36s} y={m.get('_year')} "
            f"vol={str(m.get('_entry','')).split('-')[0]:>8s} entry={m.get('_entry')}\n"
            f"        ctx:   {ctx or '(none)'}\n"
            f"        attrs: {attrs or '(none)'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    idx, seen = {}, collections.Counter()
    for m in M:
        k = (m["_entry"], str(m["_local_id"]))
        seen[k] += 1
        idx[k + (seen[k],)] = m

    A, B = clusters(a.control), clusters(a.treatment)
    bi = {m: i for i, c in enumerate(B) for m in c}

    # Control clusters that the treatment broke apart.
    split = [c for c in A if len({bi[m] for m in c}) > 1]
    print(f"{len(split)} control clusters were split by the conflict term "
          f"(showing up to {a.limit})\n")

    roles = collections.Counter()
    shown = 0
    for c in sorted(split, key=lambda s: (-len(s), sorted(s))):
        pieces = collections.defaultdict(list)
        for m in c:
            pieces[bi[m]].append(m)
        # Report the pair that actually carries the conflict.
        keys = sorted(pieces)
        rep = None
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                for mx in pieces[keys[x]]:
                    for my in pieces[keys[y]]:
                        r = score(idx[mx], idx[my], stats, geo=geo, vol_of=vol_of)
                        hits = [l for l, _ in r["terms"] if "conflict" in l]
                        if hits:
                            rep = (idx[mx], idx[my], r, hits)
                            break
                    if rep: break
                if rep: break
            if rep: break
        if not rep:
            continue
        mx, my, r, hits = rep
        for h in hits:
            for role in ("parent", "spouse", "enslaver"):
                if f"conflict:{role}" in h:
                    roles[role] += 1
        shown += 1
        if shown > a.limit:
            continue
        print(f"--- split #{shown}: cluster of {len(c)} mentions -> "
              f"{len(pieces)} identities " + "-" * 20)
        print(describe(mx))
        print(describe(my))
        print(f"      score: {r['log_odds']:+.2f}  ({explain(r)})")
        print()

    print(f"\nconflict role that caused the split ({shown} splits with a "
          f"scorable conflicting pair):")
    for role, n in roles.most_common():
        print(f"    {role:10s} {n:4d}")
    if shown > a.limit:
        print(f"\n  {shown - a.limit} further splits not printed (limit {a.limit}).")


if __name__ == "__main__":
    main()
