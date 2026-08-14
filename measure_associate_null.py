"""What is a shared associate actually worth? Measure it instead of deriving it.

THE CLAIM UNDER TEST
--------------------
`network_llr` weights a shared associate at `-ln(p)`, the name's own rarity. That
is the log-likelihood ratio ONLY under the null that the two records are
unrelated strangers, for whom naming the same person happens at the base rate p.

HANDOVER sec.14 argues that null is wrong for a parish register: the realistic
alternative to "same person" is "two DIFFERENT but related people", and relatives
share associates constantly. Under that null the evidence is worth far less.
The symptom: of 2,148 known-different pairs the model would auto-merge, 2,139
have exactly ONE shared associate, and the role pairs are godchild/godchild,
child/child, witness/witness -- co-participants at one event.

That argument has never been measured. This measures it.

    empirical LLR = ln[ P(share | same person) / P(share | different people) ]

WHY TWO NEGATIVE CLASSES, NOT ONE
---------------------------------
The obvious negative set -- two people in one entry -- is the WRONG null on its
own, and using it alone would over-correct. Those pairs are co-participants by
construction: a husband and wife at one baptism, two witnesses to one marriage.
They share associates at a rate no cross-entry candidate pair ever would, so
P(share | different) estimated on them is an upper bound, not an estimate.

So a second, better-matched negative class:

    BOTH-SACRAMENT-PRINCIPALS, CROSS-ENTRY.
    You are baptized once, born once, buried once. Two mentions that are each
    the principal of a once-in-a-lifetime sacrament of the SAME type, in
    DIFFERENT entries, cannot be one person. The merge already vetoes them, so
    they cost nothing to label -- and unlike within-entry pairs they are drawn
    from the same cross-entry population the scorer actually meets.

Reporting both bounds the answer instead of pretending to a point estimate.

POSITIVES
---------
Records transcribed twice. Mentions of the same local id across the two copies
are the same person, unarguably. Few, and easy by construction -- they agree on
everything -- so they bound P(share | same) from ABOVE too. Stated, not hidden.

WHAT THIS DOES NOT DO
---------------------
It changes no weight and writes nothing under a locked path. It is a
measurement, and the decision it informs is Daniel's, not mine.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats


def _assoc(m):
    """Flat set of associate names, the same view network_llr takes."""
    na = E._assoc_names(m)
    return set().union(*na.values()) if na else set()


def _roles_of(m, name):
    return {r for r, s in E._assoc_names(m).items() if name in s}


def _pair_stats(x, y, stats):
    """(n_shared, max_rarity_of_shared, same_role_flag) for one pair."""
    ax, ay = _assoc(x), _assoc(y)
    if not ax or not ay:
        return None                      # absence of evidence; scorer returns 0
    shared = ax & ay
    if not shared:
        return (0, 0.0, False)
    best = max(stats.llr(n) for n in shared)
    same_role = any(_roles_of(x, n) & _roles_of(y, n) for n in shared)
    return (len(shared), best, same_role)


def _summarise(label, rows, out):
    """rows: list of (n_shared, best_rarity, same_role)."""
    n = len(rows)
    if not n:
        print(f"  {label:<34} (none)")
        return {}
    shared_any = sum(1 for r in rows if r[0] > 0)
    shared_one = sum(1 for r in rows if r[0] == 1)
    shared_two = sum(1 for r in rows if r[0] >= 2)
    same_role = sum(1 for r in rows if r[2])
    rec = {"pairs": n,
           "p_share_any": shared_any / n,
           "p_share_exactly_one": shared_one / n,
           "p_share_two_plus": shared_two / n,
           "share_same_role": same_role / n}
    print(f"  {label:<34} pairs={n:>9,}  P(share>=1)={rec['p_share_any']:.4f}"
          f"  P(=1)={rec['p_share_exactly_one']:.4f}"
          f"  P(>=2)={rec['p_share_two_plus']:.4f}")
    out[label] = rec
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped_sided")
    ap.add_argument("--out", default=None,
                    help="where to write the json; defaults to stdout only. "
                         "Do NOT point this under a locked label path.")
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {a.assembled!r}")
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    print(f"corpus: {len(entries):,} entries, {len(M):,} mentions "
          f"({os.path.basename(a.assembled)})\n")

    by_entry = collections.defaultdict(list)
    for m in M:
        by_entry[m["_entry"]].append(m)

    out = {"corpus": a.assembled, "entries": len(entries), "mentions": len(M)}

    # ---- negative class 1: within-entry (co-participants; UPPER bound) ----
    within = []
    for people in by_entry.values():
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                r = _pair_stats(people[i], people[j], stats)
                if r:
                    within.append(r)

    # ---- negative class 2: cross-entry both-sacrament-principals ----
    #  Same once-in-a-lifetime sacrament type, different entries => different
    #  people. Drawn from the population the scorer actually meets.
    prin = collections.defaultdict(list)
    for m in M:
        if not m.get("_unique_sacrament"):
            continue
        for sac in (m.get("_sacraments") or ()):
            if sac in ("baptism", "birth", "burial"):
                prin[sac].append(m)
    cross = []
    CAP = 400_000                       # keep it tractable; declared below
    capped = False
    for sac, people in prin.items():
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                if people[i]["_entry"] == people[j]["_entry"]:
                    continue
                r = _pair_stats(people[i], people[j], stats)
                if r:
                    cross.append(r)
                if len(cross) >= CAP:
                    capped = True
                    break
            if capped:
                break
        if capped:
            break

    # ---- positives: the same record transcribed twice ----
    seen, dupes = {}, []
    for eid, people in by_entry.items():
        key = json.dumps(sorted((str(p.get("name") or ""),
                                 str(p.get("_local_id"))) for p in people),
                         ensure_ascii=False)
        if len(people) < 1:
            continue
        if key in seen and seen[key] != eid:
            a_people = {p["_local_id"]: p for p in by_entry[seen[key]]}
            for p in people:
                q = a_people.get(p["_local_id"])
                if q is None:
                    continue
                r = _pair_stats(p, q, stats)
                if r:
                    dupes.append(r)
        else:
            seen.setdefault(key, eid)

    print("SHARING RATES")
    neg1 = _summarise("NEG within-entry (co-participant)", within, out)
    neg2 = _summarise("NEG cross-entry both-principals", cross, out)
    pos = _summarise("POS same record twice", dupes, out)
    if capped:
        print(f"  (cross-entry class capped at {CAP:,} pairs -- "
              f"a rate, not a total)")
        out["cross_entry_capped_at"] = CAP

    # ---- the number the model uses, versus the number the corpus says ----
    print("\nWHAT ONE SHARED ASSOCIATE IS WORTH")
    print("  model: -ln(p) of the shared name, capped at "
          f"{E.MAX_LLR_PER_ASSOCIATE:.1f} nats\n")
    if pos and neg1 and neg2:
        for label, neg in (("vs co-participant null", neg1),
                           ("vs cross-entry null   ", neg2)):
            pn, qn = pos["p_share_any"], neg["p_share_any"]
            if qn > 0 and pn > 0:
                llr = math.log(pn / qn)
                print(f"  empirical LLR {label}: {llr:+.2f} nats"
                      f"   (P_same={pn:.4f} / P_diff={qn:.4f})")
                out[f"empirical_llr_{label.strip().replace(' ', '_')}"] = llr

        mean_model = sum(min(r[1], E.MAX_LLR_PER_ASSOCIATE)
                         for r in within if r[0] > 0)
        k = sum(1 for r in within if r[0] > 0)
        if k:
            print(f"\n  mean weight the model actually assigns a sharing "
                  f"known-different pair: +{mean_model / k:.2f} nats")
            out["model_mean_weight_on_sharing_negatives"] = mean_model / k

    print("\nREAD THIS BEFORE QUOTING ANY OF IT")
    print("  Both negative classes are guaranteed-different but neither is a")
    print("  random sample of candidates. Within-entry pairs are the worst")
    print("  case and cross-entry principals the better-matched one, so the")
    print("  truth sits between the two LLRs, not at either.")
    print("  Positives agree on everything by construction, so P(share|same)")
    print("  is an upper bound too -- which pushes both LLRs UP, not down.")

    if a.out:
        if "blocked_labels" in a.out or "conflict_labels" in a.out \
                or "targeted" in a.out:
            raise SystemExit(f"refusing to write under a locked label path: {a.out}")
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
