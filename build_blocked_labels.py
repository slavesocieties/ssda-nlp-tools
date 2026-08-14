"""Sample the pairs blocking throws away, so recall can be measured at all.

THE ONLY BLIND SPOT WE CANNOT SEE INTO
--------------------------------------
Every measurement in this pipeline is a form of precision: we look at merges we
made and ask whether they were right. A true merge that `_shares_context`
discarded is invisible -- never scored, never reviewed, never in a label set,
never in an A/B. 3,013,215 pairs sit in that state, having survived every other
guard.

WHY THIS IS NOT A UNIFORM SAMPLE, AND WHY THAT IS STILL UNBIASED
----------------------------------------------------------------
Almost all of those pairs are hopeless -- a near-name match 90 years apart. Two
hundred uniformly drawn rows would be two hundred zeros and would teach nothing.
But simply hand-picking the promising ones would make the resulting rate
meaningless, which is the trap: the interesting sample and the measurable sample
pull in opposite directions.

So the pairs are STRATIFIED, each stratum is sampled at its own rate, and every
sampled pair carries `weight = stratum_size / sampled_from_stratum`. A grade of
"same person" on a row drawn from a stratum of 900,000 at a rate of 1-in-30,000
stands for 30,000 pairs. Summing weight x (grade/100) over the graded rows is
then a Horvitz-Thompson estimate of how many true merges the blocked set hides,
and it stays unbiased no matter how hard the informative strata are
over-sampled -- as long as EVERY stratum keeps a non-zero rate, which is
enforced below.

Sampling is a per-stratum reservoir (Algorithm R) in a single pass, so the
3M-pair enumeration never has to be held in memory.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random

import ssda_nlp_tools.disambiguate as D
from ssda_nlp_tools import evidence as E
from ssda_nlp_tools.textmatch import name_similarity, phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo
from build_targeted_labels import render, require_corpus

YEAR_WINDOW = 60


def rarity_of(x, y, stats):
    """UNCAPPED name rarity, in nats.

    `NameStats.llr` caps at MAX_NAME_LLR (5.5) because that is what the scorer
    is allowed to spend. Using it here made every pair "common" -- no name can
    exceed 5.5, so thresholds above it are unreachable and the whole rarity
    dimension silently collapsed to one value. Stratifying is not scoring, and
    wants the real quantity: 85% of names in this corpus sit above 8.0 nats.
    """
    return min(-math.log(stats.p(x.get("name") or "")),
               -math.log(stats.p(y.get("name") or "")))


def stratum_of(x, y, stats, cuts):
    """Coarse, and deliberately about PLAUSIBILITY rather than about the model.

    Nothing here consults the scorer. If the strata were defined by what the
    model already believes, the grades would come back confirming it.
    """
    sim = name_similarity(x.get("name"), y.get("name"))
    nm = "exact" if sim >= 0.999 else ("close" if sim >= 0.90 else "loose")
    rarity = rarity_of(x, y, stats)
    r = "rare" if rarity >= cuts[1] else ("mid" if rarity >= cuts[0] else "common")
    gap = abs(x["_year"] - y["_year"])
    g = "60-74" if gap < 75 else ("75-89" if gap < 90 else "90+")
    ctx = "both-have-context" if (x.get("_ctx") and y.get("_ctx")) else "thin"
    return f"{nm}|{r}|{g}|{ctx}"


# How many rows each stratum is worth showing. Plausible strata are worth more
# of Daniel's attention; the rest still get a floor so the estimator stays
# unbiased. These are SHARES, normalised against the budget below.
def priority(st):
    nm, r, g, ctx = st.split("|")
    p = 1.0
    p *= {"exact": 12.0, "close": 3.0, "loose": 0.35}[nm]
    p *= {"rare": 4.0, "mid": 1.5, "common": 0.5}[r]
    p *= {"60-74": 2.5, "75-89": 1.2, "90+": 0.4}[g]
    p *= {"both-have-context": 1.6, "thin": 0.7}[ctx]
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped_sided")
    ap.add_argument("--outdir", default="production/luna_v3/blocked_labels")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--per-stratum-cap", type=int, default=60,
                    help="reservoir depth; the sample is drawn from these")
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(os.path.join(a.assembled, "*.materialized.json"))):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    require_corpus(entries, a.assembled)
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = E.NameStats(M, is_clergy=E._clergy)
    geo = load_geo()

    # Rarity cuts from the corpus itself, so the bands stay meaningful if the
    # name distribution changes rather than being three numbers I picked.
    rr = sorted(-math.log(stats.p(m.get("name") or "")) for m in M if m.get("name"))
    cuts = (rr[len(rr) // 3], rr[2 * len(rr) // 3])
    print(f"rarity cuts (uncapped nats): mid>={cuts[0]:.2f} rare>={cuts[1]:.2f}")

    blocks = collections.defaultdict(list)
    for m in M:
        blocks[phonetic_key(m.get("name"))].append(m)

    rng = random.Random(a.seed)
    size = collections.Counter()          # true stratum sizes, for the weights
    res = collections.defaultdict(list)   # per-stratum reservoir
    seen = collections.Counter()

    for key, g in blocks.items():
        if not key or len(g) < 2:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                x, y = g[i], g[j]
                if x["_entry"] == y["_entry"]:
                    continue
                if D._shares_context(x, y, YEAR_WINDOW):
                    continue              # not blocked; the merger saw it
                # Only pairs NOTHING else would have stopped. A pair the lifespan
                # veto would refuse anyway is not a loss and must not dilute the
                # denominator.
                if D.lifespan_conflict(x, y):
                    continue
                if x.get("_unique_sacrament") and y.get("_unique_sacrament"):
                    continue
                if E._clergy(x) or E._clergy(y):
                    continue              # Daniel: clergy are a special case
                st = stratum_of(x, y, stats, cuts)
                size[st] += 1
                seen[st] += 1
                # Algorithm R
                if len(res[st]) < a.per_stratum_cap:
                    res[st].append((x, y))
                else:
                    k = rng.randrange(seen[st])
                    if k < a.per_stratum_cap:
                        res[st][k] = (x, y)

    total = sum(size.values())
    print(f"{total:,} blocked pairs that no other guard would have stopped, "
          f"in {len(size)} strata")
    if not total:
        raise SystemExit("nothing to sample -- refusing to write an empty set")

    # Allocate the budget by priority, but never zero: every stratum with any
    # pairs gets at least one row, or its share of the total is unestimable.
    # A stratum's floor rises with its size. With one row standing for 538,000
    # pairs, a single ungenerous grade moves the estimate by half a million --
    # unbiased but useless. Three rows cuts that leverage to a third, which is
    # the difference between "we cannot say" and a usable bound. The estimate is
    # still driven by the plausible strata; this only stops the heavy ones from
    # being decided by a coin flip.
    def floor(n):
        return 4 if n >= 300_000 else 3 if n >= 100_000 else 2 if n >= 20_000 else 1

    keys = sorted(size)
    w = {k: priority(k) * size[k] ** 0.5 for k in keys}
    tot_w = sum(w.values())
    alloc = {}
    for k in keys:
        alloc[k] = max(floor(size[k]),
                       min(len(res[k]), int(round(a.size * w[k] / tot_w))))
        alloc[k] = min(alloc[k], len(res[k]))
    # trim/grow to hit the budget exactly, largest strata first
    while sum(alloc.values()) > a.size:
        k = max((k for k in keys if alloc[k] > floor(size[k])),
                key=lambda k: alloc[k], default=None)
        if k is None:
            break
        alloc[k] -= 1
    while sum(alloc.values()) < a.size:
        k = max((k for k in keys if alloc[k] < len(res[k])),
                key=lambda k: size[k] / alloc[k], default=None)
        if k is None:
            break
        alloc[k] += 1

    rows, meta = [], []
    # HEAVIEST STRATUM FIRST, so that grading top-down buys the most population
    # per judgement. `keys` is sorted by stratum NAME, which scatters weight
    # arbitrarily through the page -- and the page never shows the grader what a
    # row is worth, so there is no way to tell from the outside.
    #
    # Measured on the 2026-08-10 draw, which Daniel graded top-down: the first
    # 20 rows stand for 1,995 pairs, 0.066% of the 3,013,215 pool. The 20
    # heaviest stand for 1,692,576 -- 56.2%, and 848x more. Seventeen rows cover
    # half the pool. A partial grading is the NORMAL case for a busy supervisor,
    # so the order in which rows are offered decides what a partial grading is
    # worth.
    for k in sorted(keys, key=lambda k: -(size[k] / alloc[k])):
        picks = rng.sample(res[k], alloc[k])
        weight = size[k] / alloc[k]       # inverse sampling probability
        for x, y in picks:
            gap = abs(x["_year"] - y["_year"])
            rows.append({"a": x, "b": y,
                         "d": {"shared": 0,
                               "na": len({n for _, n in (x.get("_ctx") or ())}),
                               "nb": len({n for _, n in (y.get("_ctx") or ())}),
                               "gap": gap}})
            meta.append({"id": f"blk-{len(meta):03d}", "stratum": k,
                         "stratum_size": size[k], "sampled": alloc[k],
                         "weight": round(weight, 2),
                         "a": {"entry": x["_entry"], "id": x["_local_id"],
                               "name": x.get("name")},
                         "b": {"entry": y["_entry"], "id": y["_local_id"],
                               "name": y.get("name")}})

    print(f"drew {len(meta)} rows across {len(alloc)} strata; "
          f"weights {min(m['weight'] for m in meta):,.0f}"
          f"..{max(m['weight'] for m in meta):,.0f}")
    print(f"\n{'stratum':40s} {'size':>10} {'drawn':>6} {'weight':>10}")
    for k in sorted(keys, key=lambda k: -size[k])[:14]:
        print(f"  {k:38s} {size[k]:10,} {alloc[k]:6d} {size[k]/alloc[k]:10,.0f}")

    blurb = ("These pairs were <b>never scored at all</b>. Our pre-filter drops a pair "
             "when the two records come from different registers, name nobody in "
             "common, and sit more than 60 years apart &mdash; on the assumption that "
             "no such pair could be one person. That assumption has never been "
             "checked, and if it is wrong the merges it loses are invisible to every "
             "other measurement we have. Pairs that were impossible anyway (lifespan, "
             "two once-in-a-lifetime sacraments) are already removed, as is clergy. "
             "<b>Most of these should be obvious zeros</b> &mdash; that is the expected "
             "answer and a useful one. Same scale: <b>0</b> certainly different, "
             "<b>100</b> certainly the same."
             "<br><br><b>Rows are ordered by how much each one stands for, heaviest "
             "first.</b> They are a weighted sample, so they are not equally "
             "informative: the top rows each represent tens of thousands of "
             "discarded pairs and the bottom ones represent a handful. "
             "<b>If you only have time for some, do them from the top</b> &mdash; "
             "the first 17 rows cover half the discarded pool, and the first 20 "
             "cover 56% of it.")
    os.makedirs(a.outdir, exist_ok=True)
    hp = os.path.join(a.outdir, "blocked_pairs.html")
    open(hp, "w", encoding="utf-8").write(
        render(rows, geo, tag="blocked_by_prefilter",
               title="Never scored: pairs the pre-filter discarded",
               blurb=blurb, store="blk", filename="blocked_labels.json"))
    jp = os.path.join(a.outdir, "blocked_pairs.json")
    json.dump({"seed": a.seed, "population": total,
               "year_window": YEAR_WINDOW, "pairs": meta},
              open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {hp}\n-> {jp}")
    print(f"""
  To read the grades back: each row carries `weight`, the number of blocked
  pairs it stands for. sum(weight * grade/100) estimates how many true merges
  the {total:,} unscored pairs contain. Report a range, not a point -- with 200
  rows against 3M pairs a single "same person" on a heavy row moves the estimate
  by tens of thousands.""")


if __name__ == "__main__":
    main()
