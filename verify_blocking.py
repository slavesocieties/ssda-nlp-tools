"""Does the scalable blocking lose merges? Measure, do not assume.

The new scheme (ssda_nlp_tools/blocking.py) generates candidates from keys
instead of enumerating every same-name pair and filtering afterwards. It is only
worth adopting if it keeps the pairs that matter, so this compares it against
the current behaviour on the same corpus and reports what it drops -- BY WHAT
THE DROPPED PAIR WOULD HAVE DONE, not by how many there are.

A dropped pair that scores below the review bar costs nothing. A dropped pair
that would have auto-merged is a lost person. Those must be counted separately;
a single number over both would hide the only one that matters.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import time

import ssda_nlp_tools.blocking as B
import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, score
from ssda_nlp_tools.textmatch import phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--max-block", type=int, default=B.DEFAULT_MAX_BLOCK)
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {a.assembled!r}")
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]
    print(f"{len(M):,} mentions\n")

    rep = B.block_size_report(M, a.max_block)
    print("new blocking, key sizes only (no pair enumeration):")
    for k, v in rep.items():
        print(f"    {k:26s} {v:,}")

    # --- the current candidate set --------------------------------------
    t0 = time.time()
    old = set()
    blocks = collections.defaultdict(list)
    for i, m in enumerate(M):
        blocks[phonetic_key(m.get("name"))].append(i)
    old_enumerated = 0
    for key, idxs in blocks.items():
        if not key:
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                i, j = idxs[x], idxs[y]
                old_enumerated += 1
                if M[i]["_entry"] == M[j]["_entry"]:
                    continue
                if not D._shares_context(M[i], M[j], 60):
                    continue
                old.add((min(i, j), max(i, j)))
    t_old = time.time() - t0
    print(f"\ncurrent : enumerated {old_enumerated:,} pairs, kept {len(old):,} "
          f"({t_old:.0f}s)")

    # --- the new candidate set -------------------------------------------
    t0 = time.time()
    st = collections.Counter()
    new = set()
    for i, j in B.candidate_pairs(M, a.max_block, stats=st):
        if M[i]["_entry"] == M[j]["_entry"]:
            continue
        new.add((min(i, j), max(i, j)))
    t_new = time.time() - t0
    print(f"new     : generated {len(new):,} pairs directly ({t_new:.0f}s)")
    if st:
        print(f"          skipped {st['skipped_oversized_keys']:,} oversized keys "
              f"({st['skipped_pairs']:,} pairs)")
    print(f"\n  enumeration avoided: {old_enumerated - len(new):,} pairs "
          f"({100*(1 - len(new)/max(old_enumerated,1)):.1f}% fewer than the current "
          f"scan)")

    # --- what is lost, weighted by what it would have done ---------------
    lost = old - new
    gained = new - old
    print(f"\n  in current but NOT new : {len(lost):,}")
    print(f"  in new but NOT current : {len(gained):,}  (over-generation, harmless)")

    verdict = collections.Counter()
    examples = []
    for (i, j) in lost:
        x, y = M[i], M[j]
        if D.lifespan_conflict(x, y):
            verdict["would have been lifespan-vetoed"] += 1
            continue
        r = score(x, y, stats, geo=geo, vol_of=vol_of)
        if r["vetoed"]:
            verdict[f"would have been vetoed ({r['vetoed']})"] += 1
        elif r["log_odds"] >= E.AUTO_MERGE_LOG_ODDS:
            verdict["WOULD HAVE AUTO-MERGED"] += 1
            if len(examples) < 8:
                examples.append((x, y, r["log_odds"]))
        elif r["log_odds"] >= E.REVIEW_LOG_ODDS:
            verdict["would have gone to review"] += 1
        else:
            verdict["would have been refused anyway"] += 1

    print(f"\n  what the {len(lost):,} dropped pairs would have done:")
    for k, v in verdict.most_common():
        flag = "   <-- REAL LOSS" if "AUTO-MERGE" in k else ""
        print(f"      {k:44s} {v:9,}{flag}")

    lost_merges = verdict["WOULD HAVE AUTO-MERGED"]
    lost_review = verdict["would have gone to review"]
    print(f"\n  VERDICT")
    print(f"      merges lost : {lost_merges:,}")
    print(f"      review lost : {lost_review:,}")
    if examples:
        print(f"\n  lost merges (inspect these before adopting):")
        for x, y, lo in examples:
            print(f"      {lo:+6.2f}  {str(x.get('name'))[:28]:30s} "
                  f"{x['_entry']} y={x.get('_year')}")
            print(f"              {str(y.get('name'))[:28]:30s} "
                  f"{y['_entry']} y={y.get('_year')}")
    if not lost_merges:
        print("\n  No merge is lost. The dropped pairs are ones the scorer refuses,")
        print("  which is the outcome Daniel's ruling predicts: a common bare name")
        print("  from another register sharing nobody cannot support a merge.")


if __name__ == "__main__":
    main()
