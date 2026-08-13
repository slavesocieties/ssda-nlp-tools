"""Measure precision without asking a historian anything.

Daniel cannot label everything, and 25-35 graded pairs cannot calibrate a dozen
weights. But the corpus already contains labels nobody had to write, because the
structure of a sacramental register asserts identity and difference by itself.

GUARANTEED NEGATIVES: two people in ONE entry.
    The extractor separated them, and the merge stage already treats that as an
    impossibility -- `same-entry` is a veto. So every within-entry pair is a
    known-different pair, for free, in bulk.

    They are also the HARDEST possible negatives, which is what makes them
    worth using. A same-entry pair shares the register, the date, the parish and
    usually several associates; every circumstantial term in the model fires in
    favour of merging them. The only thing separating them is name evidence. If
    the model can hold these apart it can hold apart anything easier, and the
    rate at which it fails is a genuine false-positive bound.

GUARANTEED POSITIVES: a record transcribed twice.
    59 entries in this corpus are byte-identical to another entry. The people in
    them are literally the same people, so the corresponding mentions across the
    two copies are known-same pairs. Small, but free and unarguable.

WHAT THIS BUYS
    A precision estimate and a defensible place to put the thresholds, computed
    from the corpus rather than chosen. `AUTO_MERGE_LOG_ODDS = 3.0` is currently
    a prior; the negative distribution says what it should be for a stated
    false-positive rate.

WHAT IT DOES NOT BUY
    Recall. Same-entry negatives say nothing about merges we are failing to
    make; that is what the blocked-pair sample is for. And these negatives are
    not a random sample of the candidate pool -- they are deliberately the worst
    case -- so the false-positive rate here is an UPPER bound on the pool, not
    an estimate of it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, score
from ssda_nlp_tools.textmatch import name_similarity
from ssda_nlp_tools.volume_geo import load as load_geo


def _score_ignoring_same_entry(x, y, stats, geo, vol_of):
    """Score as if the two mentions came from different entries.

    The same-entry veto would return -inf and tell us nothing about how the
    EVIDENCE reads. Only `_entry` is altered, and only on a copy; nothing else
    in the scorer consults it.
    """
    y2 = dict(y)
    y2["_entry"] = str(y.get("_entry")) + "\x00sameentry"
    return score(x, y2, stats, geo=geo, vol_of=vol_of)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--out", default="production/luna_v3/structural_calibration.json")
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

    by_entry = collections.defaultdict(list)
    for m in M:
        by_entry[m["_entry"]].append(m)

    # ---------- guaranteed negatives -----------------------------------
    neg, neg_scored = [], 0
    vetoed_other = collections.Counter()
    for eid, people in by_entry.items():
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                x, y = people[i], people[j]
                r = _score_ignoring_same_entry(x, y, stats, geo, vol_of)
                if r["vetoed"]:
                    vetoed_other[r["vetoed"]] += 1
                    continue
                neg_scored += 1
                neg.append((r["log_odds"], x, y))

    print(f"GUARANTEED NEGATIVES (two people in one entry)")
    print(f"  pairs                    : {neg_scored + sum(vetoed_other.values()):,}")
    print(f"  refused by another veto  : {dict(vetoed_other)}")
    print(f"  actually scored          : {neg_scored:,}")
    if not neg_scored:
        raise SystemExit("no scorable within-entry pairs -- nothing to calibrate")

    los = sorted(r for r, _, _ in neg)
    def pct(p):
        return los[min(len(los) - 1, int(p * len(los)))]
    print(f"  log-odds distribution    : p50={pct(.50):+.2f} p90={pct(.90):+.2f} "
          f"p99={pct(.99):+.2f} p99.9={pct(.999):+.2f} max={los[-1]:+.2f}")

    above_review = sum(1 for r in los if r >= E.REVIEW_LOG_ODDS)
    above_auto = sum(1 for r in los if r >= E.AUTO_MERGE_LOG_ODDS)
    print(f"\n  KNOWN-DIFFERENT pairs the model would send to review: "
          f"{above_review:,}  ({100*above_review/neg_scored:.3f}%)")
    print(f"  KNOWN-DIFFERENT pairs the model would AUTO-MERGE    : "
          f"{above_auto:,}  ({100*above_auto/neg_scored:.3f}%)")
    print("  (the same-entry veto catches these in production; this measures")
    print("   what the EVIDENCE says, which is what generalises to other pairs)")

    for label, target in (("1 in 1,000", 0.999), ("1 in 10,000", 0.9999)):
        print(f"  threshold for a {label} false-merge rate on these: "
              f"{pct(target):+.2f}")

    worst = sorted(neg, key=lambda t: -t[0])[:8]
    print(f"\n  worst known-different pairs (evidence most wrongly in favour):")
    for lo, x, y in worst:
        print(f"    {lo:+7.2f}  {str(x.get('name'))[:26]:28s} vs "
              f"{str(y.get('name'))[:26]:28s}  {x['_entry']}")

    # ---------- guaranteed positives ------------------------------------
    # Byte-identical entries are one record transcribed twice, so a person in
    # copy A and the same person in copy B are the same human being.
    groups = collections.defaultdict(list)
    for e in entries:
        eid = str(e.get("entry") or e.get("id") or "")
        data = e.get("data") or {}
        if not (data.get("people") or []):
            continue                       # empty payloads all hash alike
        h = hashlib.sha1(json.dumps(data, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()
        groups[(eid.split("-")[0], h)].append(eid)

    pos = []
    for (_vol, _h), eids in groups.items():
        if len(eids) < 2:
            continue
        a_people = {m["_local_id"]: m for m in by_entry.get(eids[0], [])}
        for other in eids[1:]:
            for m in by_entry.get(other, []):
                twin = a_people.get(m["_local_id"])
                if twin is None:
                    continue
                r = _score_ignoring_same_entry(twin, m, stats, geo, vol_of)
                if not r["vetoed"]:
                    pos.append((r["log_odds"], twin, m))

    print(f"\nGUARANTEED POSITIVES (one record transcribed twice)")
    print(f"  pairs scored             : {len(pos):,}")
    if pos:
        pl = sorted(r for r, _, _ in pos)
        n_auto = sum(1 for r in pl if r >= E.AUTO_MERGE_LOG_ODDS)
        n_rev = sum(1 for r in pl if r >= E.REVIEW_LOG_ODDS)
        print(f"  log-odds                 : p10={pl[len(pl)//10]:+.2f} "
              f"p50={pl[len(pl)//2]:+.2f} max={pl[-1]:+.2f}")
        print(f"  reach auto-merge         : {n_auto:,}/{len(pl):,} "
              f"({100*n_auto/len(pl):.1f}%)")
        print(f"  reach review             : {n_rev:,}/{len(pl):,} "
              f"({100*n_rev/len(pl):.1f}%)")
        print("  NOTE: these share a name AND every attribute by construction,")
        print("  so this is an easiest-case check, not a recall estimate.")

    out = {
        "negatives": {"scored": neg_scored,
                      "p50": pct(.50), "p90": pct(.90), "p99": pct(.99),
                      "p999": pct(.999), "max": los[-1],
                      "above_review": above_review, "above_auto": above_auto},
        "positives": {"scored": len(pos)},
        "thresholds_now": {"auto": E.AUTO_MERGE_LOG_ODDS,
                           "review": E.REVIEW_LOG_ODDS},
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
