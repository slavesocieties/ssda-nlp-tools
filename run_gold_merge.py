#!/usr/bin/env python3
"""run_gold_merge.py — our merge scorer against SSDA's hand-labelled pairs.

Offline, $0, no network, no key.

    python run_gold_merge.py --gold ../ssda-openai/disambiguate.json

`disambiguate.json` in slavesocieties/openai carries 59 pairs a person labelled
match true/false. That is the same judgement Daniel is about to make 1,000 times,
so it is worth running our scorer against it BEFORE those labels arrive: if the
scorer is wrong in some systematic way, better to find it now than to discover it
after a thousand labels have been collected against it.

WHAT THIS CAN AND CANNOT MEASURE, because the two are very different here
------------------------------------------------------------------------
It is 55 positives and 4 negatives, so no accuracy figure computed over the
whole set means anything: predicting "merge" every time scores 93%.

  THE 4 NEGATIVES ARE THE REAL TEST, and they are pointed straight at Daniel's
  ruling. Every one is a same-name pair -- two Juanas, two Thomases, two Tomas
  Angel Josephs, two Unknowns -- that a person judged to be different people.
  "No people should be merged strictly based on name correspondence" is exactly
  what these four encode. Any of them merging is a precision bug.

  THE 55 POSITIVES ARE NOT A FAIR RECALL TEST. The gold person records carry
  name, titles, occupation and relationships, but no events and therefore no
  DATES. Date overlap is one of the four corroborating signals our scorer
  counts, so it is structurally unavailable here and the scorer will under-merge
  on this data in a way it would not on ours. Recall below is reported for
  completeness and should not be read as a quality number.

Also, 23 of the 59 pairs match a mention against an already-merged CLUSTER (the
`id` field is a list). Those are steps in an incremental clustering trace, not
independent pair judgements, and they are counted separately.
"""
import argparse
import json
from collections import Counter

from ssda_nlp_tools.disambiguate import (MIN_SIGNALS_FOR_ANY_MERGE,
                                         corroborating_signals, pair_score,
                                         surname_tier_allows)
from ssda_nlp_tools.textmatch import normalize_name

AUTO = 0.86
REVIEW = 0.70


def entry_of(pid):
    """`0009-02-P01` -> `0009-02`. A list means an already-merged cluster."""
    if isinstance(pid, list):
        return entry_of(pid[0]) if pid else ""
    return "-".join(str(pid).split("-")[:-1])


def to_mention(person, names, register):
    """Gold person record -> the mention shape the scorer expects."""
    ctx = set()
    for r in person.get("relationships") or []:
        if isinstance(r, dict):
            rn = names.get(str(r.get("related_person")))
            rt = r.get("relationship_type")
            if rn and rt:
                ctx.add((str(rt).lower(), rn))
    m = dict(person)
    m["_entry"] = entry_of(person.get("id"))
    m["_local_id"] = str(person.get("id"))
    m["_ctx"] = ctx
    m["_unique_sacrament"] = False      # no events in the gold records
    # All 59 pairs come from one volume. Splitting the entry id would make each
    # PAGE its own register and silently remove the shared-register signal.
    m["_register"] = register
    m["_year"] = None                   # no events, so no date overlap available
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="../ssda-openai/disambiguate.json")
    ap.add_argument("--register", default="166470")
    ap.add_argument("--out", default="production/luna_v3/gold_merge.json")
    args = ap.parse_args(argv)

    gold = json.load(open(args.gold, encoding="utf-8"))["manual"]

    names = {}
    for pair in gold:
        for p in pair["people"]:
            if not isinstance(p.get("id"), list):
                names[str(p["id"])] = normalize_name(p.get("name"))

    rows = []
    for pair in gold:
        a, b = (to_mention(p, names, args.register) for p in pair["people"])
        score, reasons = pair_score(a, b)
        allowed, tier = surname_tier_allows(a, b)
        signals = corroborating_signals(a, b)
        merged = allowed and score >= AUTO and len(signals) >= MIN_SIGNALS_FOR_ANY_MERGE
        rows.append({
            "expected": bool(pair.get("match")),
            "predicted_merge": bool(merged),
            "score": round(score, 3),
            "signals": signals,
            "surname_tier": tier,
            "blocked": not allowed,
            "cluster_step": any(isinstance(p.get("id"), list) for p in pair["people"]),
            "names": [p.get("name") for p in pair["people"]],
            "ids": [p.get("id") for p in pair["people"]],
            "reasons": reasons,
        })

    neg = [r for r in rows if not r["expected"]]
    pos = [r for r in rows if r["expected"]]
    atomic_pos = [r for r in pos if not r["cluster_step"]]

    print(f"gold pairs: {len(rows)}  ({len(pos)} match, {len(neg)} no-match)")
    print(f"of which incremental cluster steps rather than atomic pairs: "
          f"{sum(1 for r in rows if r['cluster_step'])}\n")

    print("=== THE TEST THAT MATTERS: 4 hand-labelled NON-matches ===")
    print("    (every one is a same-name pair, which is Daniel's ruling exactly)")
    bad = 0
    for r in neg:
        ok = not r["predicted_merge"]
        bad += not ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {r['names'][0]!r} vs {r['names'][1]!r}")
        print(f"           score {r['score']:.3f}  signals {r['signals']}  "
              f"tier {r['surname_tier']}  blocked={r['blocked']}")
    print(f"\n    precision on the negatives: {len(neg)-bad}/{len(neg)}"
          f"{'  <-- A FAILURE HERE IS A REAL BUG' if bad else '  (none merged)'}")

    caught = sum(1 for r in atomic_pos if r["predicted_merge"])
    print(f"\n=== recall on {len(atomic_pos)} atomic positives: {caught} "
          f"({100*caught/max(len(atomic_pos),1):.0f}%) ===")
    print("    NOT a quality number. The gold records carry no events, so no")
    print("    dates, so date overlap -- one of the four signals a merge needs --")
    print("    cannot fire at all on this data. Under-merging here is expected.")
    why = Counter()
    for r in atomic_pos:
        if not r["predicted_merge"]:
            why["blocked by surname tier" if r["blocked"]
                else f"only {len(r['signals'])} corroborating signal(s)"] += 1
    for k, v in why.most_common():
        print(f"      {v:3d}  {k}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"negatives_held": len(neg) - bad, "negatives": len(neg),
                   "atomic_positive_recall": [caught, len(atomic_pos)],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
