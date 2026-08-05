#!/usr/bin/env python3
"""verify_label_scores.py — re-score Daniel's labelled pairs against current code.

Offline, $0, no network, no key.

    python verify_label_scores.py
    python verify_label_scores.py --labels labels.json --show 12

WHY THIS IS NOT A THREE-LINE SCRIPT
-----------------------------------
`labels.json` stores POINTERS, not people:

    {"a": {"entry": "176899-0025-B-02", "id": "P02"}, ...}

The obvious harness -- hand the stored dicts straight to pair_score -- returns
0.00 for every pair, because those dicts have no "name" and name_similarity(None,
None) short-circuits. That failure is silent and it flatters: it reported all ten
of Daniel's negatives as "now correctly refused", which is exactly the answer I
wanted to see. The control that caught it is the one below, and it is the reason
this file exists rather than a scratch snippet.

So the pointers are resolved through the SAME mention builder the pipeline uses
(`_mentions_from_volume`), keyed the SAME way (`_entry`, `_local_id`), and scored
with the SAME relationship context (`_ctx`). Reimplementing any of those by hand
measures the reimplementation, not the pipeline.

THE CONTROL, which runs first and hard-fails:
  * pairs Daniel labelled 100 must not all collapse to one value
  * pairs stored at >=0.90 must still score high on average
A degenerate harness fails both, and fails them before printing any verdict.
"""
import argparse
import glob
import json
import os
import sys

from ssda_nlp_tools.disambiguate import _mentions_from_volume, pair_score

AUTO_THRESHOLD = 0.86


def load_mentions(assembled):
    paths = sorted(glob.glob(os.path.join(assembled, "*.materialized.json")))
    if not paths:
        raise SystemExit(f"no *.materialized.json under {assembled}")
    entries = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        entries.extend(d.get("entries") or d.get("records") or [])
    mentions = _mentions_from_volume({"id": "corpus", "entries": entries})
    return {(str(m["_entry"]), str(m["_local_id"])): m for m in mentions}, len(paths)


def merged_lookup(path):
    """(entry, local id) -> identity index, so "did we merge them" is read off
    the delivered run rather than re-derived."""
    if not os.path.exists(path):
        return None
    out = {}
    for k, ident in enumerate(json.load(open(path, encoding="utf-8"))):
        for m in ident.get("mentions") or []:
            out[(str(m.get("entry")), str(m.get("id")))] = k
    return out


def score(idx, pair):
    a = idx.get((str(pair["a"]["entry"]), str(pair["a"]["id"])))
    b = idx.get((str(pair["b"]["entry"]), str(pair["b"]["id"])))
    if a is None or b is None:
        return None, ["unresolved"]
    return pair_score(a, b, a.get("_ctx"), b.get("_ctx"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--identities",
                    default="production/luna_v3/merge/v8lifespan.identities.json",
                    help="the delivered run whose outcomes we are checking")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args(argv)

    idx, nvol = load_mentions(args.assembled)
    print(f"{len(idx):,} mentions from {nvol} volumes")

    pairs = json.load(open(args.labels, encoding="utf-8"))["labels"]
    labelled = [x for x in pairs if x.get("likelihood") is not None]
    unres = [x for x in pairs if score(idx, x)[0] is None]
    print(f"{len(pairs):,} pairs, {len(labelled)} labelled, {len(unres)} unresolvable")
    if unres:
        print(f"  ! {len(unres)} pointers do not resolve; results are partial")

    # ---- CONTROL, before any verdict -------------------------------------
    print("\nCONTROL (a degenerate harness fails here, not silently later)")
    # capped at 300: this is the CONTROL sample, not a reported result.
    # Nothing is withheld from any verdict below.
    hi = [x for x in pairs if x.get("score", 0) >= 0.90][:300]
    hi_now = [s for s, _ in (score(idx, x) for x in hi) if s is not None]
    pos = [x for x in labelled if x["likelihood"] == 100]
    pos_now = [s for s, _ in (score(idx, x) for x in pos) if s is not None]
    spread = len(set(round(s, 2) for s in hi_now))
    mean_hi = sum(hi_now) / len(hi_now) if hi_now else 0.0
    print(f"  {len(hi_now)} pairs stored >=0.90 -> mean now {mean_hi:.2f}, "
          f"{spread} distinct values")
    if pos_now:
        print(f"  {len(pos_now)} pairs Daniel labelled 100 -> mean now "
              f"{sum(pos_now) / len(pos_now):.2f}")
    ok = spread > 1 and mean_hi >= 0.5
    print(f"  harness {'OK' if ok else 'DEGENERATE'}")
    if not ok:
        print("\n  Refusing to report verdicts from a harness that cannot "
              "reproduce known-high scores.")
        return 1

    # ---- score is NOT the verdict ----------------------------------------
    # A pair can score 1.00 and still be refused by the surname tier, the
    # sacrament-principal guard, the attribute guard or the chronology guard,
    # all of which run AFTER scoring. Reading the score alone said 8 of Daniel's
    # 10 negatives "still merge"; the pipeline actually merged 1. Report what
    # the run did, and keep the score only as the explanation.
    ident = merged_lookup(args.identities)
    if ident is None:
        print(f"\n  no identities at {args.identities}; scores only, no verdicts")
        return 0

    def did_merge(x):
        a = ident.get((str(x["a"]["entry"]), str(x["a"]["id"])))
        b = ident.get((str(x["b"]["entry"]), str(x["b"]["id"])))
        return None if a is None or b is None else a == b

    print(f"\nWHAT THE RUN ACTUALLY DID ({os.path.basename(args.identities)})")
    print(f"  {'names':44s} {'label':>5} {'score':>6}  outcome")
    rows = []
    for x in sorted(labelled, key=lambda y: y["likelihood"]):
        s, reasons = score(idx, x)
        m = did_merge(x)
        rows.append((x["likelihood"], m))
        flag = "" if x["likelihood"] not in (0, 100) else (
            "  <-- DISAGREES" if (x["likelihood"] == 100) != m else "")
        print(f"  {' / '.join(x['names'])[:44]:44s} {x['likelihood']:5} {s:6.2f}"
              f"  {'merged' if m else 'split'}{flag}")
        if args.show and flag:
            print(f"  {'':44s}              {'; '.join(reasons[:3])}")

    hard = [r for r in rows if r[0] in (0, 100)]
    ok = sum(1 for L, m in hard if (L == 100) == m)
    print(f"\n  AGREEMENT on the {len(hard)} pairs he was certain about: "
          f"{ok}/{len(hard)} ({100 * ok / len(hard):.0f}%)")
    print(f"    negatives he wanted split : "
          f"{sum(1 for L, m in hard if L == 0 and not m)}/"
          f"{sum(1 for L, _ in hard if L == 0)}")
    print(f"    positives he wanted merged: "
          f"{sum(1 for L, m in hard if L == 100 and m)}/"
          f"{sum(1 for L, _ in hard if L == 100)}")
    print("\n  25 labels is a small sample; treat this as a smoke test with a "
          "real signal, not a benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
