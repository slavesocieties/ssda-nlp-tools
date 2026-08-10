"""Build a label set that can actually settle the conflicting-relationship weights.

WHY A SECOND SET EXISTS
-----------------------
Daniel ruled on 2026-08-07 that two same-named people with different spouses are
"essentially disqualifying". The term is implemented, but of the 25 pairs he has
graded, exactly ONE contains a conflicting relationship. His labels cannot
referee this ruling -- not because they are poor labels, but because they were
sampled on same-name/same-parish, and a conflict is rare within that.

So this samples pairs BECAUSE they conflict, stratified by which role conflicts,
how far apart the two events are, and -- most importantly -- whether the conflict
is the only thing keeping them apart.

THE STRATUM THAT MATTERS
------------------------
`flips` marks pairs that WOULD have reached the review bar without the conflict
term and do not reach it with the term. Those are the pairs where the ruling is
actually doing work; everything else would have been refused anyway. They are
deliberately over-sampled, and the page says so, because a label on a pair whose
outcome the term does not change teaches nothing about the term.

Weights are the point of this set, so it does NOT filter by score: including only
pairs the current model likes would return labels that confirm the current model.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

import ssda_nlp_tools.disambiguate as D
from ssda_nlp_tools import evidence as E
from ssda_nlp_tools.textmatch import phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo
from build_targeted_labels import render, require_corpus

MAX_BLOCK = 400


def conflicts_of(a, b, stats):
    _, why = E.network_llr(a, b, stats)
    return [w for w in why if w.startswith("conflict")]


def gather(M, stats, vol_of, geo):
    blocks = collections.defaultdict(list)
    for m in M:
        blocks[phonetic_key(m.get("name"))].append(m)

    out = []
    dropped = collections.Counter()
    for idxs in blocks.values():
        if len(idxs) < 2 or len(idxs) > MAX_BLOCK:
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                a, b = idxs[x], idxs[y]
                if a["_entry"] == b["_entry"]:
                    continue
                if E._clergy(a) or E._clergy(b):
                    dropped["clergy"] += 1
                    continue
                if D.lifespan_conflict(a, b):
                    dropped["lifespan-impossible"] += 1
                    continue
                if a.get("_unique_sacrament") and b.get("_unique_sacrament"):
                    dropped["both-sacrament-principals"] += 1
                    continue
                hits = conflicts_of(a, b, stats)
                if not hits:
                    continue
                on = E.score(a, b, stats, geo=geo, vol_of=vol_of)
                if on["vetoed"]:
                    dropped[f"veto-{on['vetoed']}"] += 1
                    continue
                pen = sum(float(h.split("(")[1].rstrip(")")) for h in hits)
                off_lo = on["log_odds"] - pen
                out.append((a, b, hits, on["log_odds"], off_lo))
    total = len(out) + sum(dropped.values())
    print(f"pairs with a conflicting relationship: {len(out):,} kept "
          f"of {total:,} examined")
    for k, v in dropped.most_common():
        print(f"    dropped {k:30s} {v:8,}")
    return out


def stratum(hits, on_lo, off_lo):
    roles = sorted({h.split("(")[0].split(":")[1] for h in hits})
    role = "+".join(roles)
    flips = (off_lo >= E.REVIEW_LOG_ODDS > on_lo)
    band = ("would-merge-without" if flips else
            "refused-either-way" if on_lo < E.REVIEW_LOG_ODDS else
            "passes-anyway")
    return f"{role}|{band}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--outdir", default="production/luna_v3/conflict_labels")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260810)
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(os.path.join(a.assembled, "*.materialized.json"))):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    require_corpus(entries, a.assembled)
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = E.NameStats(M, is_clergy=E._clergy)
    geo = load_geo()
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    pool = gather(M, stats, vol_of, geo)
    if not pool:
        raise SystemExit("no conflicting pairs found -- refusing to write an empty set")

    strata = collections.defaultdict(list)
    for rec in pool:
        strata[stratum(rec[2], rec[3], rec[4])].append(rec)
    print(f"\n{len(strata)} strata:")
    for k in sorted(strata, key=lambda s: -len(strata[s])):
        print(f"    {k:44s} {len(strata[k]):7,}")

    # Take EVERY pair whose outcome the ruling actually changes, then water-fill
    # the remainder for contrast.
    #
    # A plain round-robin over all 15 strata would have given the decision-
    # relevant ones about a thirteenth of the page while the page told Daniel
    # they were the point of it. Grading a pair the term does not move teaches
    # nothing about how strong the term should be, so those pairs are here only
    # as a control, and the split is reported rather than hidden.
    rng = random.Random(a.seed)
    for v in strata.values():
        rng.shuffle(v)
    flip_keys = [k for k in sorted(strata) if k.endswith("|would-merge-without")]
    rest_keys = [k for k in sorted(strata) if k not in flip_keys]

    chosen = []
    for k in flip_keys:
        while strata[k] and len(chosen) < a.size:
            chosen.append((k, strata[k].pop()))
    n_flip = len(chosen)
    while len(chosen) < a.size and any(strata[k] for k in rest_keys):
        for k in rest_keys:
            if len(chosen) >= a.size:
                break
            if strata[k]:
                chosen.append((k, strata[k].pop()))
    rng.shuffle(chosen)
    print(f"\ndrew {len(chosen)} pairs across "
          f"{len({k for k, _ in chosen})} strata")
    print(f"    {n_flip} where the conflict CHANGES the outcome "
          f"(all {sum(len(strata[k]) for k in flip_keys) + n_flip} available)")
    print(f"    {len(chosen)-n_flip} controls where it does not")

    rows, meta = [], []
    for st, (x, y, hits, on_lo, off_lo) in chosen:
        gap = (abs(x["_year"] - y["_year"])
               if x.get("_year") and y.get("_year") else None)
        _, why = E.network_llr(x, y, stats)
        rows.append({"a": x, "b": y,
                     "d": {"shared": sum(1 for w in why if w.startswith("shared")),
                           "na": len({n for _, n in (x.get("_ctx") or ())}),
                           "nb": len({n for _, n in (y.get("_ctx") or ())}),
                           "gap": gap}})
        meta.append({"id": f"cfl-{len(meta):03d}", "stratum": st,
                     "conflicts": hits,
                     "a": {"entry": x["_entry"], "id": x["_local_id"],
                           "name": x.get("name")},
                     "b": {"entry": y["_entry"], "id": y["_local_id"],
                           "name": y.get("name")}})

    blurb = ("Every pair below shares a name <b>and disagrees on a relationship</b> "
             "&mdash; a different spouse, a set of parents that cannot both be right, "
             "or a different enslaver. You said such a disagreement should be "
             "essentially disqualifying; these are the cases where that judgement "
             "actually changes the outcome, so grading them is what lets us set how "
             "strong the penalty should be. Impossible pairings (lifespan, two "
             "once-in-a-lifetime sacraments) have already been removed. Scale as "
             "before: <b>0</b> certainly different, <b>100</b> certainly the same.")
    os.makedirs(a.outdir, exist_ok=True)
    html_path = os.path.join(a.outdir, "conflict_pairs.html")
    open(html_path, "w", encoding="utf-8").write(
        render(rows, geo, tag="relationship_conflict",
               title="Same name, conflicting relationship",
               blurb=blurb, store="cfl", filename="conflict_labels.json"))
    json_path = os.path.join(a.outdir, "conflict_pairs.json")
    json.dump({"seed": a.seed, "pairs": meta},
              open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {html_path}\n-> {json_path}")
    print("\nThe page does not show the model's answer, for the same reason as before.")


if __name__ == "__main__":
    main()
