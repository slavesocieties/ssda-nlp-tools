"""Prove Daniel's conflicting-relationship ruling actually fires, and measure it.

Daniel, 2026-08-07: "is it also looking for relationships that explicitly don't
match? I.e. if two people with the same name have spouses with different names,
that's essentially disqualifying - it's not just that they have no shared
relationship, they have actively distinguishing individual relationships."

Before this change a DIFFERENT spouse contributed +0.00 -- byte-identical to
knowing nothing. This script checks three things, in order of how easily each
could be faked:

  1. the term fires at all, on hand-built mentions (a no-op flag is this
     project's most-repeated bug)
  2. it does NOT fire when the two names are one person spelled two ways
  3. what it does to REAL candidate pairs, which is the only number worth
     sending anyone
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from ssda_nlp_tools import evidence
from ssda_nlp_tools.evidence import NameStats, network_llr, score


def _m(name, year=None, **ctx):
    """A mention with (role, name) context pairs."""
    return {"name": name, "_year": year,
            "_ctx": {(r, n) for r, n in ctx.items()}}


def unit_cases():
    """Hand-built pairs where the right answer is known by inspection."""
    stats = NameStats([{"name": n} for n in
                       ["maria"] * 80 + ["josefa"] * 40 + ["juan"] * 60 +
                       ["ana"] * 30 + ["ines"] * 10 + ["custodio vieira"] * 1])
    cases = [
        ("no relationships either side",
         _m("maria", 1850), _m("maria", 1852), 0.0),
        ("different SPOUSES, same decade",
         _m("maria", 1850, spouse="juan perez"),
         _m("maria", 1852, spouse="pedro gomez"), evidence.W_CONFLICT_SPOUSE),
        ("different SPOUSES, 40 years apart (remarriage plausible)",
         _m("maria", 1810, spouse="juan perez"),
         _m("maria", 1852, spouse="pedro gomez"), evidence.W_CONFLICT_SPOUSE / 3),
        # A mother in one record and a father in the other is NOT a conflict.
        # The audit of the A/B found this splitting two 1742 entries for one
        # "Maria de Jesus" who named the SAME husband in both.
        ("one parent each, different -> mother+father, NOT a conflict",
         _m("josefa", 1850, parent="ana lopez"),
         _m("josefa", 1851, parent="ines vega"), None),
        ("3 distinct parents between them -> impossible, IS a conflict",
         {"name": "josefa", "_year": 1850,
          "_ctx": {("parent", "ana lopez"), ("parent", "juan lopez")}},
         {"name": "josefa", "_year": 1851,
          "_ctx": {("parent", "ines vega")}}, evidence.W_CONFLICT_PARENT),
        ("different ENSLAVERS",
         _m("josefa", 1850, enslaver="ana lopez"),
         _m("josefa", 1851, enslaver="ines vega"), evidence.W_CONFLICT_ENSLAVER),
        ("SAME spouse spelled two ways -> must NOT be a conflict",
         _m("maria", 1850, spouse="juan perez"),
         _m("maria", 1852, spouse="juan peres"), None),
        ("same spouse (positive control, must stay positive)",
         _m("maria", 1850, spouse="custodio vieira"),
         _m("maria", 1852, spouse="custodio vieira"), None),
    ]
    print("=" * 78)
    print("1. DOES THE TERM FIRE?  (network LLR only, no name/place/date terms)")
    print("=" * 78)
    ok = True
    for label, a, b, expect in cases:
        llr, why = network_llr(a, b, stats)
        note = ""
        if expect is not None:
            hit = any("conflict" in w for w in why)
            want_hit = expect != 0.0
            if hit != want_hit:
                note, ok = "   <-- WRONG", False
        print(f"  {label:52s} {llr:+7.2f}  {','.join(why) or '-'}{note}")

    # The controls, checked explicitly rather than eyeballed.
    by_label = {c[0]: c for c in cases}
    var = by_label["SAME spouse spelled two ways -> must NOT be a conflict"]
    if any("conflict" in w for w in network_llr(var[1], var[2], stats)[1]):
        print("  FAIL: spelling variant treated as a different person"); ok = False
    same = by_label["same spouse (positive control, must stay positive)"]
    if network_llr(same[1], same[2], stats)[0] <= 0:
        print("  FAIL: a shared spouse stopped being positive evidence"); ok = False
    mf = by_label["one parent each, different -> mother+father, NOT a conflict"]
    if any("conflict" in w for w in network_llr(mf[1], mf[2], stats)[1]):
        print("  FAIL: a mother/father pair charged as a parent conflict"); ok = False
    print(f"\n  unit checks: {'PASS' if ok else 'FAIL'}")
    return ok


def real_pairs(assembled, limit_vols):
    """How often does a conflict actually occur among the pairs the pipeline scores?

    Loaded and blocked EXACTLY as run_evidence_merge.py does -- same
    `_mentions_from_volume`, same `phonetic_key` blocking -- so the denominator
    is the real candidate set and not a convenient one.
    """
    import ssda_nlp_tools.disambiguate as D
    from ssda_nlp_tools.textmatch import phonetic_key

    paths = sorted(Path(assembled).glob("*.materialized.json"))[:limit_vols]
    if not paths:
        print(f"\nNo *.materialized.json under {assembled}; skipping real data.")
        return
    entries = []
    for p in paths:
        entries.extend(json.loads(p.read_text(encoding="utf-8"))["entries"])
    if not entries:
        print(f"\nNo entries under {assembled}; skipping real data.")
        return
    mentions = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(mentions, is_clergy=evidence._clergy)

    blocks = collections.defaultdict(list)
    for m in mentions:
        blocks[phonetic_key(m.get("name"))].append(m)

    n_pairs = n_conflict = n_flip = 0
    by_role = collections.Counter()
    examples = []
    for key, group in blocks.items():
        if not key or len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.get("_entry") and a.get("_entry") == b.get("_entry"):
                    continue
                n_pairs += 1
                _, why = network_llr(a, b, stats)
                hits = [w for w in why if w.startswith("conflict")]
                if not hits:
                    continue
                n_conflict += 1
                for h in hits:
                    by_role[h.split("(")[0]] += 1
                # Did it change the DECISION, not just the score?
                # (score != disposition -- the mistake made earlier in this project)
                after = score(a, b, stats)
                if after["vetoed"]:
                    continue
                # Read the penalty ACTUALLY applied out of the reason string.
                # Recomputing it from the constants gets the decayed
                # (remarriage) spouse case wrong -- -4.0 instead of -1.33 --
                # which silently overstates how many pairs flipped.
                pen = sum(float(h.split("(")[1].rstrip(")")) for h in hits)
                before_lo = after["log_odds"] - pen
                if before_lo >= evidence.REVIEW_LOG_ODDS > after["log_odds"]:
                    n_flip += 1
                    if len(examples) < 6:
                        examples.append((a.get("name"), a, b, before_lo,
                                         after["log_odds"], hits))

    print("\n" + "=" * 78)
    print(f"2. REAL DATA  ({len(paths)} volumes, {len(entries):,} entries, "
          f"{len(mentions):,} mentions)")
    print("=" * 78)
    print(f"  phonetically-blocked candidate pairs: {n_pairs:,}")
    if not n_pairs:
        return
    print(f"  with a conflicting relationship    : {n_conflict:,} "
          f"({100*n_conflict/n_pairs:.2f}%)")
    for role, c in by_role.most_common():
        print(f"      {role:24s} {c:6,}")
    print(f"  pairs pushed BELOW the review bar  : {n_flip:,} "
          f"({100*n_flip/n_pairs:.2f}% of pairs examined)")
    print("\n  NOTE: these are PAIRS, not people, and not merges. A pair falling")
    print("  below the bar only matters if nothing else was already stopping it.")
    if examples:
        print("\n  examples:")
        for name, a, b, lo0, lo1, hits in examples:
            print(f"    {name!r:22s} {lo0:+6.2f} -> {lo1:+6.2f}  {','.join(hits)}")
            for m in (a, b):
                ctx = ", ".join(f"{r}={n}" for r, n in sorted(m.get("_ctx") or ()))
                vol = str(m.get("_entry") or "?").split("-")[0]
                print(f"        vol {vol:>8s} "
                      f"y={m.get('_year') or m.get('year')}  {ctx[:88]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--volumes", type=int, default=10 ** 6)
    a = ap.parse_args()
    good = unit_cases()
    real_pairs(a.assembled, a.volumes)
    sys.exit(0 if good else 1)
