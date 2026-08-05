#!/usr/bin/env python3
"""calibrate_evidence.py — fit the evidence weights to Daniel's labels.

Offline, $0, no network, no key.

    python calibrate_evidence.py
    python calibrate_evidence.py --l2 2.0 --report production/luna_v3/calibration.json

THE SAMPLING PROBLEM, WHICH DECIDES THE WHOLE METHOD
----------------------------------------------------
`labels.json` is NOT a random sample of candidate pairs. It is a stratified draw
over SCORED pairs, deliberately over-representing the interesting ones: 1,000
pairs standing for 7,305,667. Fitting a plain logistic regression on it and
using the result wholesale is how you get a model that merges a third of the
corpus, because the fitted intercept encodes the sample's base rate (about 50%
positive) rather than the population's (about 1 in 260).

Under choice-based (outcome-dependent) sampling the SLOPES are still consistent
and only the INTERCEPT is biased -- a standard result, and a convenient one. So:

    slopes     fitted from the labels, which is what they can support
    intercept  NOT fitted. Set from the population prior, which is a property of
               the corpus and knowable without labels.

The stratified draw also carries inverse-probability `weight` per pair, so the
regression is weighted and the estimates refer to the population rather than to
the sample.

WHAT IS DELIBERATELY EXCLUDED
  * the 25 graded SYNTHETIC labels. Every synthetic name is absent from the
    corpus, so the name-rarity feature -- the single most discriminating term --
    is constant across all of them. Fitting on data where a feature cannot vary
    teaches nothing about that feature and distorts the others to compensate.
  * pairs Daniel marked 25/50/75. Graded uncertainty is real information, but a
    binary regression cannot use it honestly; it belongs in a threshold study.

24 points against 7 features is thin, so the fit is L2-regularised toward the
hand-set priors rather than toward zero: absent evidence, a weight should stay
where a stated argument put it, not collapse.
"""
import argparse
import collections
import glob
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

import ssda_nlp_tools.disambiguate as D
from ssda_nlp_tools import evidence as E
from ssda_nlp_tools.volume_geo import load as load_geo

FEATURES = ["name", "network", "clergy", "place", "vol_disjoint", "date", "attrs"]


def featurise(a, b, stats, geo, vol_of):
    """The same terms score() uses, but as a vector with the weights factored
    out, so a fit recovers the weights themselves."""
    from ssda_nlp_tools.textmatch import name_similarity
    f = dict.fromkeys(FEATURES, 0.0)

    sim = name_similarity(a.get("name"), b.get("name"))
    rarity = min(stats.llr(a.get("name")), stats.llr(b.get("name")))
    f["name"] = sim * rarity / E.MAX_NAME_LLR          # in [0,1]

    n_llr, _ = E.network_llr(a, b, stats)
    f["network"] = n_llr / E.MAX_LLR_PER_ASSOCIATE

    f["clergy"] = 1.0 if (E._clergy(a) and E._clergy(b)) else 0.0

    if geo and vol_of:
        va, vb = vol_of(a), vol_of(b)
        lvl = geo.same_place(va, vb)
        f["place"] = {"institution": 1.0, "city": 0.6, "state": 0.2,
                      "country": 0.0, "none": -1.0}.get(lvl, 0.0)
        if geo.overlapping_years(va, vb) is False:
            f["vol_disjoint"] = 1.0

    ya, yb = a.get("_year") or a.get("year"), b.get("_year") or b.get("year")
    if ya and yb:
        gap = abs(int(ya) - int(yb))
        f["date"] = 1.0 if gap <= 20 else (-1.0 if gap > 40 else 0.0)

    agree = conflict = 0
    for k in ("phenotype", "free", "ethnicity", "origin", "occupation", "legitimate"):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        if str(x).strip().lower() == str(y).strip().lower():
            agree += 1
        else:
            conflict += 1
    f["attrs"] = 0.25 * agree - 2.5 * conflict
    return np.array([f[k] for k in FEATURES], dtype=float)


def priors():
    return np.array([E.MAX_NAME_LLR, E.MAX_LLR_PER_ASSOCIATE, E.W_CLERGY_BOTH,
                     4.0, E.W_VOLUMES_NEVER_COEXIST, E.W_YEAR_CLOSE, 1.0],
                    dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--report", default="production/luna_v3/calibration.json")
    args = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json"))):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    mentions = D._mentions_from_volume({"id": "corpus", "entries": entries})
    idx = {(str(m["_entry"]), str(m["_local_id"])): m for m in mentions}
    stats = E.NameStats(mentions, is_clergy=E._clergy)
    geo = load_geo()
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    raw = json.load(open(args.labels, encoding="utf-8"))["labels"]
    X, y, w, names = [], [], [], []
    for r in raw:
        if r.get("likelihood") not in (0, 100):
            continue
        a = idx.get((str(r["a"]["entry"]), str(r["a"]["id"])))
        b = idx.get((str(r["b"]["entry"]), str(r["b"]["id"])))
        if not a or not b:
            continue
        X.append(featurise(a, b, stats, geo, vol_of))
        y.append(1.0 if r["likelihood"] == 100 else 0.0)
        w.append(float(r.get("weight") or 1.0))
        names.append(" / ".join(r["names"])[:34])
    X = np.array(X); y = np.array(y); w = np.array(w)
    w = w / w.mean()
    print(f"{len(y)} usable labels ({int(y.sum())} positive), "
          f"{X.shape[1]} features")
    print(f"inverse-probability weights span {w.min():.2f} to {w.max():.2f}\n")

    # Intercept is FIXED at the population prior, not fitted -- see module docs.
    b0 = E.LOG_PRIOR_ODDS
    p0 = priors()

    def nll(beta):
        z = X @ beta + b0
        # weighted logistic loss + L2 pull toward the stated priors
        ll = w * (y * z - np.logaddexp(0.0, z))
        return -ll.sum() + args.l2 * float(((beta - p0) ** 2).sum())

    res = minimize(nll, p0, method="L-BFGS-B")
    beta = res.x

    print(f"{'feature':14s} {'prior':>8s} {'fitted':>8s} {'change':>9s}")
    for k, a_, b_ in zip(FEATURES, p0, beta):
        print(f"  {k:12s} {a_:8.2f} {b_:8.2f} {b_ - a_:+9.2f}")

    z = X @ beta + b0
    pred = 1 / (1 + np.exp(-z))
    acc = ((pred >= 0.5) == (y == 1)).mean()
    print(f"\nin-sample accuracy {acc:.0%} on {len(y)} labels "
          f"(in-sample, {len(y)} points, {X.shape[1]} features -- NOT a "
          f"generalisation estimate)")

    # what a bare name + weak circumstantial evidence now scores
    bare = np.zeros(len(FEATURES)); bare[FEATURES.index("name")] = 1.0
    bare_city = bare.copy()
    bare_city[FEATURES.index("place")] = 0.6
    bare_city[FEATURES.index("date")] = 1.0
    print(f"\nSTRUCTURAL CHECK (the failure the corpus A/B found)")
    for lbl, v in (("name alone", bare), ("name + same city + close date", bare_city)):
        s = float(v @ beta + b0)
        print(f"   {lbl:32s} log-odds {s:+.2f}  "
              f"{'MERGES' if s >= E.AUTO_MERGE_LOG_ODDS else 'does not merge'}")

    out = {"features": FEATURES, "priors": p0.tolist(), "fitted": beta.tolist(),
           "intercept_fixed": b0, "n_labels": len(y), "l2": args.l2,
           "in_sample_accuracy": float(acc),
           "note": "slopes fitted under choice-based sampling; intercept NOT "
                   "fitted, set from the population prior"}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump(out, open(args.report, "w", encoding="utf-8"), indent=1)
    print(f"\n-> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
