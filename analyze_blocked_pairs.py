"""What is actually inside the 8.55M pairs that blocking discards unscored?

WHY THIS MATTERS MORE THAN ANYTHING ELSE MEASURED TODAY
-------------------------------------------------------
Everything we can currently measure is a form of PRECISION. We see the merges we
made and ask whether they were right, and Daniel can grade them. There is no
measurement of RECALL anywhere in this pipeline, because a true merge that
blocking discarded never becomes a pair, never gets scored, never enters a label
set, and never appears in any A/B. It is invisible by construction.

`_shares_context` drops a pair only when ALL of these hold:
    different register, AND
    no person named in both entries, AND
    both dated, more than `year_window` (60) years apart.

So the discarded set is not arbitrary -- it is specifically long-gap,
cross-register pairs with no overlapping associate. The merges we would be
losing are long-lived people appearing in two registers decades apart.

THE DENOMINATOR HAS TO BE HONEST
--------------------------------
Blocking runs BEFORE the lifespan veto in the merge loop, so a large share of
blocked pairs are ones the pipeline would have refused anyway as
chronologically impossible. Those are not losses and must not be counted as
though they were. This separates them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats
from ssda_nlp_tools.textmatch import phonetic_key

YEAR_WINDOW = 60


def load(assembled):
    entries = []
    for p in sorted(glob.glob(f"{assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {assembled!r}")
    return D._mentions_from_volume({"id": "corpus", "entries": entries})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped")
    a = ap.parse_args(argv)

    M = load(a.assembled)
    stats = NameStats(M, is_clergy=E._clergy)
    blocks = collections.defaultdict(list)
    for m in M:
        blocks[phonetic_key(m.get("name"))].append(m)

    n_pairs = n_blocked = 0
    reason = collections.Counter()
    survives = collections.Counter()
    gap_hist = collections.Counter()
    examples = []

    for key, g in blocks.items():
        if not key or len(g) < 2:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                x, y = g[i], g[j]
                if x["_entry"] == y["_entry"]:
                    continue
                n_pairs += 1
                if D._shares_context(x, y, YEAR_WINDOW):
                    continue
                n_blocked += 1

                # Would the pipeline have refused it anyway?
                if D.lifespan_conflict(x, y):
                    reason["also lifespan-impossible"] += 1
                    continue
                if x.get("_unique_sacrament") and y.get("_unique_sacrament"):
                    reason["also both-sacrament-principals"] += 1
                    continue
                if E._clergy(x) or E._clergy(y):
                    reason["clergy (merged aggressively anyway)"] += 1
                    continue
                reason["SURVIVES every other guard"] += 1

                gap = abs(x["_year"] - y["_year"])
                gap_hist[min(gap // 10 * 10, 120)] += 1
                sim = E.name_similarity(x.get("name"), y.get("name"))
                rarity = min(stats.llr(x.get("name")), stats.llr(y.get("name")))
                survives["exact name" if sim >= 0.999 else "near name"] += 1
                if len(examples) < 12 and sim >= 0.999 and rarity >= 5.4:  # llr is capped at 5.5
                    examples.append((x, y, gap, rarity))

    print(f"in-block pairs examined : {n_pairs:,}")
    print(f"dropped by blocking     : {n_blocked:,} "
          f"({100*n_blocked/max(n_pairs,1):.1f}%)\n")
    print("of the dropped pairs, what would have happened to them anyway:")
    for k, v in reason.most_common():
        print(f"    {k:36s} {v:9,}  ({100*v/max(n_blocked,1):5.2f}%)")

    live = reason["SURVIVES every other guard"]
    print(f"\n  THE SET THAT MATTERS: {live:,} pairs that no other guard would")
    print(f"  have stopped, and that were never scored. This is the only place a")
    print(f"  true merge can be lost without leaving any trace.")
    if live:
        print(f"\n  name agreement among them: {dict(survives)}")
        print(f"  year gap distribution:")
        for g in sorted(gap_hist):
            print(f"     {g:3d}-{g+9:3d}y {gap_hist[g]:8,}")
    if examples:
        print(f"\n  the most suspicious of them (exact rare name, no shared "
              f"associate, >60y apart):")
        for x, y, gap, rarity in examples:
            print(f"    {str(x.get('name'))[:30]:32s} rarity {rarity:.1f}  gap {gap}y")
            for m in (x, y):
                ctx = ", ".join(f"{r}={n}" for r, n in sorted(m.get("_ctx") or ()))
                print(f"        {m['_entry']:22s} y={m.get('_year')} "
                      f"age={m.get('age')} {ctx[:70]}")
    print("""
  Read this the right way: a large SURVIVES count is not proof of lost merges.
  It is the size of the unexamined surface. Whether any of them are real people
  is a question only Daniel can answer, which is what the sample is for.""")


if __name__ == "__main__":
    main()
