#!/usr/bin/env python3
"""analyze_surname_tradeoff.py — what the merge bar actually costs, corpus-wide.

Offline, $0, no network, no key.

    python analyze_surname_tradeoff.py --tag v9tradeoff

WHY
---
Daniel's 25 labels produced three disagreements. I told him in a DM that two of
the splits were "the cost of the surname fix". Diagnosing them individually shows
that is wrong, and wrong in a way that matters:

  Juan Gonzalez / Juan Gonzalez Torre   surname tier ALLOWS it (distant, but 4
                                        corroborating signals clears the bar).
                                        Something later split them.
  Antonio Abad Facenda (x2)             tier REFUSES: 'exact' surname, but only
                                        1 corroborating signal where 2 are
                                        required. That is Daniel's own ruling
                                        that a name alone may not carry a merge,
                                        not the surname-symmetry change.

So the two splits have DIFFERENT causes and neither is the one I named. A guess
about which rule fired is worth nothing; this records the disposition the run
actually took, for every pair above the auto threshold.

WHAT IT MEASURES
  1. per-rule block counts -- which rule is actually expensive
  2. the Spanish double-surname case: `_surname_of` takes the LAST token, so
     "Juan Gonzalez" -> gonzalez but "Juan Gonzalez Torre" -> torre. Under the
     paternal+maternal convention those are the same family, and dropping the
     maternal surname is routine in these registers. Counts pairs where one
     name's tokens are a strict PREFIX of the other's, by disposition.
  3. the exchange rate: how many merges each rule costs, so the bar can be set
     against a number instead of an anecdote.
"""
import argparse
import collections
import json
import glob
import os

import ssda_nlp_tools.disambiguate as D


def require_corpus(entries, assembled):
    """Refuse to proceed on an empty corpus.

    Found by the review pass, and it had already done damage: pointed at an
    empty directory these tools produced empty output, reported success, and
    OVERWROTE a real artifact with it. Running the check destroyed e1 and
    v9tradeoff. An empty input is always a mistake -- a wrong --assembled path,
    an unfinished rebuild -- and never a result worth writing.
    """
    if not entries:
        raise SystemExit(
            f"no entries under {assembled!r}. Refusing to write empty output "
            f"over existing artifacts -- check the --assembled path.")
    return entries



def token_prefix(a, b):
    """True when one name is the other plus extra trailing surname tokens --
    the dropped-maternal-surname shape."""
    ta, tb = D.name_tokens(a), D.name_tokens(b)
    if not ta or not tb or len(ta) == len(tb):
        return False
    short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    return long_[:len(short)] == short


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--outdir", default="production/luna_v3/merge")
    ap.add_argument("--tag", default="v9tradeoff")
    ap.add_argument("--floor", type=float, default=0.86,
                    help="only log pairs that reached the auto threshold")
    args = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json"))):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    require_corpus(entries, args.assembled)
    print(f"{len(entries):,} entries")

    log = []
    res = D.disambiguate_volume({"id": "corpus", "entries": entries},
                                pair_log=log, pair_log_floor=args.floor,
                                collect_review=False, volume_tag=args.tag)
    print(f"{len(log):,} pairs logged at >= {args.floor}\n")

    by = collections.Counter(r["disposition"] for r in log)
    print(f"{'disposition':38s} {'pairs':>9}")
    for k, n in by.most_common():
        print(f"  {k:36s} {n:9,}")

    # The double-surname shape, by disposition.
    #
    # PAIRS ARE NOT PEOPLE, and the gap is enormous: a first cut of this reported
    # 5,824 pairs, which in the saved sample collapsed to 175 distinct NAME pairs
    # over 833 mentions, one name pair accounting for 405 rows. Quoting the pair
    # count as the scale would overstate it by an order of magnitude. All three
    # counts are reported so the number cannot be read as people.
    #
    # Placeholder surnames ("Francisco" vs "Francisco N.") are excluded: that is
    # an unrecorded name, not a dropped maternal surname, and it has its own rule.
    print(f"\nDROPPED-SURNAME SHAPE (one name is the other + extra tokens)")
    pref = collections.Counter()
    names, ments = collections.defaultdict(set), collections.defaultdict(set)
    rows = []
    skipped_placeholder = 0
    for r in log:
        na, nb = r["a"].get("name"), r["b"].get("name")
        if not token_prefix(na, nb):
            continue
        if (D.is_placeholder_surname(D._surname_of(na))
                or D.is_placeholder_surname(D._surname_of(nb))):
            skipped_placeholder += 1
            continue
        d = r["disposition"]
        pref[d] += 1
        names[d].add((na, nb))
        ments[d].add((r["a"]["entry"], r["a"]["id"]))
        ments[d].add((r["b"]["entry"], r["b"]["id"]))
        rows.append(r)
    print(f"  {sum(pref.values()):,} pair comparisons "
          f"({skipped_placeholder:,} placeholder pairs excluded)")
    print(f"  {'disposition':34s} {'pairs':>8} {'names':>7} {'mentions':>9}")
    for k, n in pref.most_common():
        print(f"    {k:32s} {n:8,} {len(names[k]):7,} {len(ments[k]):9,}")
    allm = set().union(*ments.values()) if ments else set()
    alln = set().union(*names.values()) if names else set()
    print(f"\n  {len(alln):,} distinct name pairs over {len(allm):,} distinct "
          f"mentions.\n  THAT is the scale of the question. The pair count above "
          f"is inflated by\n  repeated comparisons of the same few names and must "
          f"not be quoted as people.")
    print("  Whether these SHOULD merge is a ruling, not a bug: two men both "
          "called\n  Francisco Antonio need not be one man.")

    out = os.path.join(args.outdir, f"{args.tag}.tradeoff.json")
    json.dump({"by_disposition": dict(by),
               "dropped_surname_pairs": dict(pref),
               "dropped_surname_distinct_names": {k: len(v) for k, v in names.items()},
               "dropped_surname_distinct_mentions": {k: len(v) for k, v in ments.items()},
               "dropped_surname_total_names": len(alln),
               "dropped_surname_total_mentions": len(allm),
               "placeholder_excluded": skipped_placeholder,
               "floor": args.floor, "logged": len(log), "stats": res["stats"]},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # every refused pair, not a silent top-N: a truncated file reads as complete
    ref = [r for r in rows if r["disposition"] != "auto"]
    json.dump(ref, open(os.path.join(
        args.outdir, f"{args.tag}.dropped_surname_refused.json"), "w",
        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  {len(ref):,} refused pairs written in full (no top-N cap)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
