#!/usr/bin/env python3
"""run_evidence_merge.py — cluster the corpus with the weight-of-evidence scorer.

Offline, $0, no network, no key.

    python run_evidence_merge.py --tag e1
    python run_evidence_merge.py --tag e1-strict --auto 4.0

`evidence.py` scores a PAIR. This turns that into identities, so the model can be
compared against the delivered pipeline on the thing that actually matters --
32,783 identities over 39,697 mentions -- rather than on 24 labels.

WHAT IS DELIBERATELY REUSED FROM THE OLD PIPELINE
  * mention construction (`_mentions_from_volume`) and phonetic blocking, so the
    two runs see exactly the same candidate pairs and any difference is the
    SCORER, not the plumbing
  * the lifespan impossibility, which is a fact about people rather than a
    judgement about names

WHAT IS DELIBERATELY NOT REUSED
  * the surname tiers, the N-corroborating-signals bar, and cluster surname
    compatibility. Those are the binary gates being replaced; keeping them would
    make this a test of the gates with extra steps.

THE CLUSTER-LEVEL GUARDS ARE KEPT AS VETOES, NOT AS WEIGHTS. Union-find is
transitive, so a chain of individually-reasonable merges can still join two
clusters that share an entry or close a descent cycle. Those remain
impossibilities at the cluster level for the same reason they are at the pair
level, and they are the one place where "aggregate the evidence" does not apply:
no weight of evidence makes a person their own grandparent.
"""
import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict

import ssda_nlp_tools.disambiguate as D
from ssda_nlp_tools.evidence import (AUTO_MERGE_LOG_ODDS, REVIEW_LOG_ODDS,
                                     NameStats, _clergy, score)
from ssda_nlp_tools.textmatch import phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--outdir", default="production/luna_v3/merge")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--auto", type=float, default=AUTO_MERGE_LOG_ODDS)
    ap.add_argument("--review", type=float, default=REVIEW_LOG_ODDS)
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json")))
    entries = []
    for p in paths:
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    mentions = D._mentions_from_volume({"id": "corpus", "entries": entries})
    n = len(mentions)
    print(f"{len(paths)} volumes, {len(entries):,} entries, {n:,} mentions")

    stats = NameStats(mentions, is_clergy=_clergy)
    geo = load_geo(args.volumes)
    if geo:
        print(geo.report())
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    blocks = defaultdict(list)
    for i, m in enumerate(mentions):
        blocks[phonetic_key(m.get("name"))].append(i)

    uf = D._UnionFind(n)
    cluster_entries = {i: {m["_entry"]} for i, m in enumerate(mentions)}
    cluster_parents = defaultdict(set)
    by_local = {(m["_entry"], m["_local_id"]): k for k, m in enumerate(mentions)}
    for i, m in enumerate(mentions):
        for d in (m.get("_descendants") or ()):
            k = by_local.get((m["_entry"], str(d)))
            if k is not None:
                cluster_parents[i].add(k)

    reasons = Counter()
    auto = review = 0
    t0 = time.time()
    for idxs in blocks.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                mi, mj = mentions[i], mentions[j]
                if mi["_entry"] == mj["_entry"]:
                    continue
                if not D._shares_context(mi, mj, 60):
                    reasons["blocked-context"] += 1
                    continue
                if D.lifespan_conflict(mi, mj):
                    reasons["veto-lifespan"] += 1
                    continue
                r = score(mi, mj, stats, geo=geo, vol_of=vol_of)
                if r["vetoed"]:
                    reasons[f"veto-{r['vetoed']}"] += 1
                    continue
                lo = r["log_odds"]
                if lo < args.review:
                    reasons["below-review"] += 1
                    continue
                if lo < args.auto:
                    review += 1
                    reasons["review"] += 1
                    continue
                if D._clusters_share_an_entry(uf, i, j, cluster_entries):
                    reasons["veto-cluster-same-entry"] += 1
                    continue
                if D._would_close_ancestry_cycle(uf, i, j, cluster_parents):
                    reasons["veto-ancestry-cycle"] += 1
                    continue
                ra, rb = uf.find(i), uf.find(j)
                uf.union(i, j)
                root = uf.find(i)
                cluster_entries[root] = (cluster_entries.get(ra, set())
                                         | cluster_entries.get(rb, set()))
                cluster_parents[root] = (cluster_parents.get(ra, set())
                                         | cluster_parents.get(rb, set()))
                auto += 1
    elapsed = time.time() - t0

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    identities = []
    for k, (_, idxs) in enumerate(sorted(clusters.items()), 1):
        members = [mentions[i] for i in idxs]
        names = Counter(m.get("name") or "" for m in members)
        identities.append({
            "person_id": f"{args.tag}-{k:04d}",
            "canonical_name": max(names, key=lambda x: (names[x], len(x))),
            "n_mentions": len(idxs),
            "mentions": [{"entry": m["_entry"], "id": m["_local_id"],
                          "name": m.get("name")} for m in members],
        })

    out = {"mentions": n, "identities": len(identities),
           "merged_identities": sum(1 for i in identities if i["n_mentions"] > 1),
           "auto_merges": auto, "review_pairs": review,
           "dispositions": dict(reasons),
           "config": {"scorer": "evidence", "auto": args.auto,
                      "review": args.review, "seconds": round(elapsed, 1),
                      "volumes": [os.path.basename(p) for p in paths],
                      "geo": bool(geo)}}
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, args.tag)
    json.dump(identities, open(f"{base}.identities.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(out, open(f"{base}.stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"\n{len(identities):,} identities  ({out['merged_identities']:,} merged)")
    print(f"{auto:,} auto-merges, {review:,} review pairs")
    scored = auto + review + reasons["below-review"]
    if scored:
        print(f"review rate over scored pairs: {100 * review / scored:.2f}%")
    for k, v in reasons.most_common():
        print(f"   {k:28s} {v:12,}")
    print(f"\n{elapsed:.0f}s -> {base}.{{identities,stats}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
