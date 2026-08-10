"""Judge the conflicting-relationship term against Daniel's graded labels.

These labels EVALUATE. They cannot calibrate: 25 graded pairs do not constrain
the dozen weights in the model, and fitting to them would only prove the model
can memorise 25 things.

THE MAPPING HAZARD, WHICH IS THE WHOLE REASON THIS FILE IS CAREFUL
------------------------------------------------------------------
`targeted_labels.json` stores {"0": 0, "1": 75, ...} -- POSITIONS, with no pair
identifiers. A position only means something against the exact pair file Daniel
was looking at when he graded. The targeted set was regenerated today (with the
impossible pairs removed), so scoring the labels against the NEW file would
silently compare his judgement of one pair with the model's judgement of a
different one, and would still print a plausible-looking accuracy.

So: the OLD file is used, and before any verdict is printed, two controls run.

CONTROL 1  score must correlate positively with Daniel's grade. If the mapping
           were scrambled the correlation would sit near zero, and a near-zero
           correlation is reported as a FAILURE rather than as a finding.
CONTROL 2  the pairs must resolve. A resolver that matches nothing returns 0.00
           for everything, which previously produced a flattering "10/10
           negatives fixed".
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, explain, score
from ssda_nlp_tools.volume_geo import load as load_geo


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="production/luna_v3/targeted/targeted_pairs.json",
                    help="the pair file Daniel actually graded -- NOT the regenerated one")
    ap.add_argument("--labels", default="ssda_nlp_tools/targeted_labels.json")
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    a = ap.parse_args(argv)

    pf = json.load(open(a.pairs, encoding="utf-8"))
    pairs = pf["pairs"] if isinstance(pf, dict) else pf
    labels = json.load(open(a.labels, encoding="utf-8"))["labels"]
    print(f"pair file : {a.pairs}  ({len(pairs)} pairs)")
    print(f"labels    : {a.labels}  ({len(labels)} graded)\n")

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]
    by = {(m["_entry"], str(m["_local_id"])): m for m in M}

    rows = []
    unresolved = 0
    for k, grade in sorted(labels.items(), key=lambda kv: int(kv[0])):
        i = int(k)
        if i >= len(pairs):
            unresolved += 1
            continue
        r = pairs[i]
        x = by.get((r["a"]["entry"], str(r["a"]["id"])))
        y = by.get((r["b"]["entry"], str(r["b"]["id"])))
        if not x or not y:
            unresolved += 1
            continue
        rows.append((i, grade, x, y))

    print(f"resolved {len(rows)}/{len(labels)} graded pairs "
          f"(unresolved {unresolved})")
    if unresolved:
        print("  !! unresolved labels mean the mapping is suspect; "
              "treat everything below as UNRELIABLE")

    def scores(on: bool):
        keep = (E.W_CONFLICT_PARENT, E.W_CONFLICT_SPOUSE, E.W_CONFLICT_ENSLAVER)
        if not on:
            E.W_CONFLICT_PARENT = E.W_CONFLICT_SPOUSE = E.W_CONFLICT_ENSLAVER = 0.0
        try:
            return [score(x, y, stats, geo=geo, vol_of=vol_of) for _, _, x, y in rows]
        finally:
            (E.W_CONFLICT_PARENT, E.W_CONFLICT_SPOUSE,
             E.W_CONFLICT_ENSLAVER) = keep

    off, on = scores(False), scores(True)
    grades = [g for _, g, _, _ in rows]
    lo_off = [r["log_odds"] if r["log_odds"] > -700 else -50.0 for r in off]
    lo_on = [r["log_odds"] if r["log_odds"] > -700 else -50.0 for r in on]

    print("\nCONTROL 1 -- score must track Daniel's grade")
    rho_off, rho_on = spearman(lo_off, grades), spearman(lo_on, grades)
    print(f"  Spearman rho (conflict OFF): {rho_off:+.3f}")
    print(f"  Spearman rho (conflict ON) : {rho_on:+.3f}")
    if abs(rho_off) < 0.15:
        print("  !! FAIL: near-zero correlation. Either the position mapping is "
              "wrong or the model has no signal here. Do not quote the numbers "
              "below as evidence about the term.")
    print("\nCONTROL 2 -- the term must actually move something")
    moved = sum(1 for x, y in zip(lo_off, lo_on) if abs(x - y) > 1e-9)
    print(f"  pairs whose score changed: {moved}/{len(rows)}")
    if moved == 0:
        print("  !! FAIL: the toggle did nothing; nothing below is attributable.")

    print("\nEFFECT BY DANIEL'S GRADE  (grade 0/25 = he thinks DIFFERENT people,"
          "\n                           75/100 = he thinks SAME person)")
    print(f"  {'grade':>6} {'n':>4} {'mean log-odds off':>18} {'on':>9} {'change':>9}"
          f"  {'crossed below review':>21}")
    for g in sorted(set(grades)):
        idx = [i for i, gg in enumerate(grades) if gg == g]
        mo = statistics.fmean(lo_off[i] for i in idx)
        mn = statistics.fmean(lo_on[i] for i in idx)
        crossed = sum(1 for i in idx
                      if lo_off[i] >= E.REVIEW_LOG_ODDS > lo_on[i])
        print(f"  {g:6d} {len(idx):4d} {mo:18.2f} {mn:9.2f} {mn-mo:+9.2f} "
              f"{crossed:21d}")

    print("\n  The number that matters: a pair Daniel graded 0 or 25 that the")
    print("  term pushed below the bar is a CORRECT save; one he graded 75 or")
    print("  100 that it pushed below is a COST.")
    save = sum(1 for i, g in enumerate(grades)
               if g <= 25 and lo_off[i] >= E.REVIEW_LOG_ODDS > lo_on[i])
    cost = sum(1 for i, g in enumerate(grades)
               if g >= 75 and lo_off[i] >= E.REVIEW_LOG_ODDS > lo_on[i])
    print(f"    correct saves: {save}")
    print(f"    costs        : {cost}")

    print("\n  per-pair detail (only pairs the term moved):")
    for i, (idx, g, x, y) in enumerate(rows):
        if abs(lo_off[i] - lo_on[i]) < 1e-9:
            continue
        print(f"    tgt-{idx:03d} grade={g:3d}  {lo_off[i]:+6.2f} -> {lo_on[i]:+6.2f}"
              f"   {str(x.get('name'))[:26]:28s} vs {str(y.get('name'))[:26]}")
        print(f"        {explain(on[i]).splitlines()[0]}")

    n = len(rows)
    print(f"\n  SAMPLE SIZE: {n} graded pairs, of which {moved} moved. "
          f"This is far too small to\n  establish a rate; it is an existence "
          f"check, and should be reported as one.")


if __name__ == "__main__":
    main()
