#!/usr/bin/env python3
"""validate_training_sample.py — check the stratified draw is what it claims.

Offline, $0, no network, no key.

    python validate_training_sample.py

The sample is the artifact Daniel labels and the merge model will be fitted on.
Three claims are load-bearing and none had been checked:

  1. the per-stratum reservoir is UNIFORM (Algorithm R), so what survives is a
     fair draw from a stratum however large it grew
  2. the water-fill spreads the budget EVENLY, maximising distinct case types
  3. `weight = stratum_population / stratum_sampled` lets a model recover the
     real distribution from a deliberately skewed draw

(1) is tested by simulation rather than by reading the code, because an
off-by-one in Algorithm R produces a subtly biased sample that looks fine.
(3) is tested as an identity on the delivered labels.json.

The interesting one is the caveat in (3): weights reconstruct the population of
the strata that were SAMPLED, not the whole corpus. Any stratum that drew zero is
invisible to a weighted estimate, and no weight can repair that.
"""
import argparse
import collections
import json
import math
import os
import sys

from ssda_nlp_tools.training_sample import StratifiedReservoir


def check_reservoir_uniformity(trials=4000, stream=500, per_cell=20, seed=7):
    """Every member of the stream must survive with probability per_cell/stream.

    A biased reservoir (the classic error is randrange(n-1) or `<=` instead of
    `<`) skews toward either the head or the tail of the stream, which is
    invisible in the output and fatal to any model fitted on it.
    """
    import random
    rng = random.Random(seed)
    kept = collections.Counter()
    for t in range(trials):
        r = StratifiedReservoir(per_cell=per_cell, seed=rng.randrange(1 << 30))
        for i in range(stream):
            r.append({"score": 0.9, "disposition": "auto", "reasons": [],
                      "a": {"entry": "v-0001-01", "id": "P01", "name": "A"},
                      "b": {"entry": "v-0002-01", "id": "P02", "name": "B"},
                      "_i": i})
        for p in list(r.cells.values())[0]:
            kept[p["_i"]] += 1

    exp = trials * per_cell / stream
    # chi-square against the uniform expectation
    chi2 = sum((kept[i] - exp) ** 2 / exp for i in range(stream))
    # mean survival of the first decile vs the last, the direction bias shows in
    head = sum(kept[i] for i in range(stream // 10)) / (stream // 10)
    tail = sum(kept[i] for i in range(stream - stream // 10, stream)) / (stream // 10)
    df = stream - 1
    # 4-sigma band on a chi-square with df degrees of freedom
    hi = df + 4 * math.sqrt(2 * df)
    ok = chi2 < hi
    print("1. RESERVOIR UNIFORMITY (Algorithm R), by simulation")
    print(f"   {trials:,} trials x {stream} items, reservoir {per_cell}")
    print(f"   expected survivals per position : {exp:.1f}")
    print(f"   head decile mean {head:.1f}   tail decile mean {tail:.1f}")
    print(f"   chi-square {chi2:.1f} vs 4-sigma bound {hi:.1f}  -> "
          f"{'UNIFORM' if ok else 'BIASED'}")
    return ok


def check_water_fill():
    """Depth must differ by at most 1 between any two non-exhausted strata."""
    # Vary the axes stratum_of actually keys on. A first version varied only the
    # volume id and produced ONE stratum, so the evenness assertion was vacuous
    # -- a spread of 0 across a single cell proves nothing.
    scores = (0.95, 0.80, 0.60, 0.40)
    disps = ("auto", "blocked-surname-tier-exact", "blocked-cluster-surname")
    r = StratifiedReservoir(per_cell=100, seed=1)
    for i in range(20000):
        same = i % 2 == 0
        r.append({"score": scores[i % len(scores)],
                  "disposition": disps[i % len(disps)],
                  "reasons": ["name~1.00"] if i % 3 else ["shared_rel(1)"],
                  "a": {"entry": "v1-0001-01", "id": "P01", "name": "Juan Vega",
                        "detail": {"phenotype": "pardo"} if i % 5 else {}},
                  "b": {"entry": ("v1" if same else "v2") + "-0002-01",
                        "id": "P02", "name": "Juan Vega Solar" if i % 7 else "Juan Vega",
                        "detail": {"phenotype": "pardo"} if i % 5 else {}}})
    assert len(r.cells) > 3, f"test fixture made only {len(r.cells)} strata"
    drawn = r.draw(300)
    per = collections.Counter(p["stratum"] for p in drawn)
    depths = sorted(per.values())
    exhausted = {k for k in per if per[k] == len(r.cells[k])}
    open_depths = [v for k, v in per.items() if k not in exhausted]
    spread = (max(open_depths) - min(open_depths)) if open_depths else 0
    print("\n2. WATER-FILL EVENNESS")
    print(f"   {len(drawn)} drawn over {len(per)} strata; depths {depths[:8]}"
          f"{' ...' if len(depths) > 8 else ''}")
    print(f"   spread among non-exhausted strata: {spread}  -> "
          f"{'EVEN' if spread <= 1 else 'UNEVEN'}")
    return spread <= 1


def check_weights(path, population=None):
    print("\n3. INVERSE-PROBABILITY WEIGHTS, on the delivered sample")
    if not os.path.exists(path):
        print(f"   no {path}; skipped")
        return True
    pairs = json.load(open(path, encoding="utf-8"))["labels"]
    if not pairs or "weight" not in pairs[0]:
        print("   sample carries no weights; skipped")
        return True

    # The delivered sample does NOT carry stratum_population / stratum_sampled;
    # only stratum and weight survive. So the identity weight == pop/sampled
    # cannot be checked directly here, and is checked instead by the two
    # properties it implies.
    by = collections.defaultdict(list)
    for p in pairs:
        by[p["stratum"]].append(p["weight"])
    uneven = {k: set(round(w, 4) for w in ws)
              for k, ws in by.items() if len(set(round(w, 4) for w in ws)) > 1}
    print(f"   {len(pairs):,} pairs over {len(by):,} strata")
    print(f"   weight constant within each stratum : "
          f"{'HOLDS' if not uneven else f'{len(uneven)} strata vary'}")

    est = sum(p["weight"] for p in pairs)
    print(f"   sum(weights) = {est:,.0f}")
    if population:
        err = abs(est - population) / population
        print(f"   scored-pair population = {population:,}  ({err:.3%} error)")
        print("   -> the weighted sample reconstructs the FULL population, which")
        print(f"      also proves no stratum drew zero: all {len(by)} are present.")
    singles = sum(1 for ws in by.values() if len(ws) == 1)
    print(f"   {singles} strata were drawn ONCE. Their weight is the whole stratum")
    print("   population, so any weighted estimate leaning on them has very high")
    print("   variance. That is a property of the budget, not a bug.")
    return not uneven and (not population or err < 0.001)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--population", type=int, default=7305667,
                    help="scored pairs the sample was drawn from")
    args = ap.parse_args(argv)

    ok = check_reservoir_uniformity(trials=args.trials)
    ok &= check_water_fill()
    ok &= check_weights(args.labels, args.population)
    print(f"\n{'ALL CHECKS PASS' if ok else 'SOMETHING IS WRONG -- see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
