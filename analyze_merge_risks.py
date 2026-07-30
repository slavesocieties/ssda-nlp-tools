#!/usr/bin/env python3
"""analyze_merge_risks.py — put numbers on the two merging questions.

Offline, $0, no network, no key. Reads the resolved identities produced by
run_pipeline / run_merge; re-runs nothing.

Both questions were put to Daniel impressionistically and both deserve better,
because every quantity guessed on this project has been wrong and every quantity
measured has held.

**Q1, the yeismo bridges.** `phonetic_fold` maps both `ll` and `y` to `i`, which
is right for pronunciation and too strong for surname identity: it makes Llepiz
look 0.80 similar to Yepez, and Llopez 0.80 similar to Lopez. In one cluster
that pulled in 51 mentions. The question Daniel actually has to answer is how
much of the corpus rests on that rule, so this counts every identity holding a
surname pair that is compatible ONLY because of it.

**Q2, given-name-only people.** The surname tiers exempt single-token names by
design -- enslaved people are routinely recorded by given name alone, and
blocking those would undo real linking. But squeezing the surname cases has
concentrated the remaining over-merging there, and every top hub in the graph is
now a common Maria. This measures how much of the graph those identities carry,
so "never auto-merge them" can be costed rather than argued.

    python analyze_merge_risks.py
"""
import argparse
import json
import re
import unicodedata
from collections import Counter

from ssda_nlp_tools.textmatch import name_tokens, phonetic_fold
from ssda_nlp_tools.disambiguate import _surname_of, surname_affinity, SURNAME_TIERS

NEAR = next(lo for lo, _, lab in SURNAME_TIERS if lab == "near")


def strict_fold(s: str) -> str:
    """`phonetic_fold` with the yeismo collapse disabled.

    Implemented by substituting sentinels BEFORE folding, so `ll` and `y` come
    out distinct while every other rule (z/s, b/v, silent h, doubled letters) is
    untouched -- a one-rule change rather than a reimplementation.

    The sentinels must be DIGITS. The first version used the letters L and Y,
    which `phonetic_fold` lowercases and then rewrites by its own rules, so both
    Llopez and Lopez collapsed to `lopes` and the diagnostic reported the pair as
    MORE similar without the rule than with it. Digits fall through the fold's
    else-branch untouched and are unaffected by lowercasing.
    """
    t = unicodedata.normalize("NFKD", str(s or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return phonetic_fold(t.replace("ll", "1").replace("y", "2"))


def strict_affinity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a or not b or a == b:
        return 1.0
    fa, fb = strict_fold(a), strict_fold(b)
    if fa == fb:
        return 1.0
    return SequenceMatcher(None, fa, fb, autojunk=False).ratio()


def surnames_of(ident):
    out = set()
    for m in ident.get("mentions") or []:
        s = _surname_of(m.get("name"))
        if s:
            out.add(s)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index",
                    default="production/luna_v3/corpus_pipeline_v4/person_index.json")
    ap.add_argument("--network",
                    default="production/luna_v3/corpus_pipeline_v4/network.json")
    args = ap.parse_args(argv)

    idents = json.load(open(args.index, encoding="utf-8"))
    total_mentions = sum(i.get("n_mentions", 0) for i in idents)
    print(f"{len(idents):,} identities, {total_mentions:,} mentions\n")

    # ---------------------------------------------------------------- Q1 ---
    print("=" * 62)
    print("Q1  identities held together only by the ll/y (yeismo) collapse")
    print("=" * 62)
    affected, aff_mentions, examples = 0, 0, []
    pair_counts = Counter()
    for ident in idents:
        sns = surnames_of(ident)
        if len(sns) < 2:
            continue
        dependent = []
        for a in sns:
            for b in sns:
                if a >= b:
                    continue
                cur = surname_affinity(f"X {a}", f"X {b}")
                strict = strict_affinity(a, b)
                # compatible now, would fall below the "near" bar without the rule
                if cur >= NEAR and strict < NEAR:
                    dependent.append((a, b, round(cur, 3), round(strict, 3)))
        if dependent:
            affected += 1
            aff_mentions += ident.get("n_mentions", 0)
            for a, b, *_ in dependent:
                pair_counts[(a, b)] += 1
            if len(examples) < 8:
                examples.append((ident["person_id"], ident.get("canonical_name"),
                                 ident.get("n_mentions"), dependent[:3]))
    print(f"identities affected : {affected:,} of {len(idents):,} "
          f"({100*affected/len(idents):.2f}%)")
    print(f"mentions inside them: {aff_mentions:,} of {total_mentions:,} "
          f"({100*aff_mentions/total_mentions:.2f}%)")
    print(f"distinct surname pairs relying on the rule: {len(pair_counts):,}\n")
    if pair_counts:
        print("  most common dependent pairs:")
        for (a, b), n in pair_counts.most_common(10):
            print(f"    {a} ~ {b}   in {n} identity(ies)")
    if examples:
        print("\n  examples:")
        for pid, name, n, dep in examples:
            print(f"    {pid}  {name!r}  x{n}")
            for a, b, cur, strict in dep:
                print(f"        {a} ~ {b}: {cur} now, {strict} without the rule")

    # ---------------------------------------------------------------- Q2 ---
    print("\n" + "=" * 62)
    print("Q2  identities with no surname on ANY mention")
    print("=" * 62)
    nameless, nameless_mentions = [], 0
    for ident in idents:
        if surnames_of(ident):
            continue
        nameless.append(ident)
        nameless_mentions += ident.get("n_mentions", 0)
    merged = [i for i in nameless if i.get("n_mentions", 0) > 1]
    absorbed = sum(i["n_mentions"] for i in merged)
    print(f"identities            : {len(nameless):,} of {len(idents):,} "
          f"({100*len(nameless)/len(idents):.1f}%)")
    print(f"mentions              : {nameless_mentions:,} of {total_mentions:,} "
          f"({100*nameless_mentions/total_mentions:.1f}%)")
    print(f"of those, MERGED (>1 mention): {len(merged):,} identities "
          f"holding {absorbed:,} mentions")
    if merged:
        print(f"\n  if given-name-only people were never auto-merged, those "
              f"{len(merged):,}\n  identities would split back into {absorbed:,} "
              f"separate people: {len(idents):,} -> "
              f"{len(idents) - len(merged) + absorbed:,} (+{absorbed-len(merged):,})")
        top = sorted(merged, key=lambda i: -i["n_mentions"])[:10]
        print("\n  largest given-name-only identities:")
        for i in top:
            xc = "cross-volume" if i.get("cross_chunk") else "one volume"
            print(f"    {i['n_mentions']:4d}  {i.get('canonical_name')!r:38s} {xc}")

    # graph weight of those identities
    try:
        net = json.load(open(args.network, encoding="utf-8"))
        deg = Counter()
        for e in net.get("edges") or []:
            deg[e.get("source")] += 1
            deg[e.get("target")] += 1
        ids = {i["person_id"] for i in merged}
        carried = sum(d for k, d in deg.items() if k in ids)
        tot = sum(deg.values())
        print(f"\n  graph weight: those identities carry {carried:,} of {tot:,} "
              f"edge endpoints ({100*carried/tot:.1f}%)")
    except (OSError, json.JSONDecodeError, ZeroDivisionError) as exc:
        print(f"\n  (graph weight unavailable: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
