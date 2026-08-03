#!/usr/bin/env python3
"""verify_claims.py — recompute every load-bearing figure we have sent Daniel.

Offline, $0, no network, no key.

    python verify_claims.py
    python verify_claims.py --dms production/luna_v3

Nine DM drafts carry ~306 numeric tokens between them. Most are dates and entry
ids; a few dozen are claims about the data that Daniel is entitled to assume
somebody checked. This recomputes those from the artifacts and reports any that
no longer match.

WHY THIS EXISTS
---------------
Twice in one day I reported a figure that was wrong in the same way: computed
once by a bespoke script, quoted from memory afterwards, and never recomputed
against the artifact.

  * "info-rich share 0.95% -> 12.5%" -- 0.95% was a different statistic
    (five attributes on BOTH sides, counting `context`). Like for like it was
    1.7%, so the improvement was ~7x, not ~13x.
  * "descent 2-cycles 28 -> 0" -- that count looked only at 2-cycles and
    double-counted ordered pairs. Total ancestry cycles went 28 -> 16, and some
    were rerouted into LONGER loops rather than removed.

Neither was a lie and both were wrong, which is the point: a number that is not
recomputed decays silently as the corpus changes underneath it.

TWO FAILURE MODES, KEPT SEPARATE
--------------------------------
  WRONG   the artifact never supported this figure
  STALE   it was right when written and the corpus has moved since

Stale is expected and fine -- 5,226 records became 5,226 minus withdrawals, the
graph was rebuilt three times today. What matters is knowing which is which
before quoting a figure again.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys


def _load_json(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def recompute(root, transcriptions, manual):
    """Every figure below is derived here and nowhere else."""
    v = {}

    # --- corpus -----------------------------------------------------------
    assembled = sorted(glob.glob(os.path.join(
        root, "production/luna_v3/assembled/*.materialized.json")))
    entries = []
    for p in assembled:
        entries.extend((_load_json(p) or {}).get("entries") or [])
    v["delivered_records"] = len(entries)
    v["delivered_volumes"] = len(assembled)

    # --- the 232-volume transcription set ---------------------------------
    tpaths = sorted(glob.glob(os.path.join(transcriptions, "*.json")))
    pages = nonempty = failed = 0
    failed_vols = set()
    marker = re.compile(r"\[TRANSCRIPTION FAILED", re.I)
    for p in tpaths:
        vol = os.path.basename(p)[:-5]
        for item in (_load_json(p) or []):
            if not isinstance(item, dict):
                continue
            pages += 1
            t = item.get("transcription") or ""
            if t.strip():
                nonempty += 1
            if marker.search(t):
                failed += 1
                failed_vols.add(vol)
    v["transcribed_volumes"] = len(tpaths)
    v["total_pages"] = pages
    v["nonempty_transcriptions"] = nonempty
    v["failed_marker_pages"] = failed
    v["failed_marker_volumes"] = len(failed_vols)
    v["failed_marker_pct"] = round(100 * failed / max(pages, 1), 2)

    # --- hand transcriptions ---------------------------------------------
    mpaths = sorted(glob.glob(os.path.join(manual, "*.json")))
    v["hand_volumes"] = len(mpaths)
    v["hand_entries"] = sum(len((_load_json(p) or {}).get("entries") or [])
                            for p in mpaths)
    v["overlap_volumes"] = len({os.path.basename(p)[:-5] for p in mpaths}
                               & {os.path.basename(p)[:-5] for p in tpaths})

    # --- transcription accuracy ------------------------------------------
    g = _load_json(os.path.join(root, "production/luna_v3/manual_gold.json")) or {}
    o = g.get("overall") or {}
    if o:
        v["gold_pages"] = o.get("pages")
        v["gold_human_chars"] = o.get("human_chars")
        v["sub_rate_pct"] = round(100 * o.get("sub_rate", 0), 2)
        v["del_rate_pct"] = round(100 * o.get("del_rate", 0), 2)
        v["median_similarity"] = o.get("median_similarity")

    # --- segmentation vs human -------------------------------------------
    s = _load_json(os.path.join(root, "production/luna_v3/seg_gold.json")) or {}
    if s:
        hum = sum(r.get("human_entries", 0) for r in s.values())
        pred = sum(r.get("predicted_entries", 0) for r in s.values())
        v["seg_human_entries"] = hum
        v["seg_found_pct"] = round(100 * pred / max(hum, 1), 1)

    # --- graph ------------------------------------------------------------
    n = _load_json(os.path.join(
        root, "production/luna_v3/corpus_final_pipeline/network.json")) or {}
    if n:
        v["graph_people"] = len(n.get("nodes") or [])
        v["graph_edges"] = len(n.get("edges") or [])
    gv = _load_json(os.path.join(root, "production/luna_v3/graph_validation.json")) or {}
    if gv:
        v["graph_self_loops"] = len(gv.get("self_loop") or [])
        v["graph_role_contradictions"] = len(gv.get("contradictory_roles") or [])
        v["graph_ancestry_cycles"] = len(gv.get("ancestry_cycle") or [])

    # --- pair sample ------------------------------------------------------
    c = _load_json(os.path.join(
        root, "production/luna_v3/training_set/pairs_daniel1k.coverage.json")) or {}
    if c:
        v["pairs_scored"] = c.get("pairs_scored")
        v["pair_strata"] = c.get("strata_present")
        v["pair_singletons"] = c.get("singleton_strata")
        v["pairs_sampled"] = c.get("sampled")

    # --- spend ------------------------------------------------------------
    led = _load_json(os.path.join(
        root, "production/luna_live/spend_ledger.json")) or {}
    if led:
        v["ledger_cap_usd"] = led.get("cap_usd")
        v["ledger_confirmed_usd"] = round(led.get("confirmed_usd", 0), 2)
        v["ledger_headroom_usd"] = round(
            led.get("cap_usd", 0) - led.get("confirmed_usd", 0)
            - led.get("reserved_usd", 0), 2)
        if v["delivered_records"]:
            v["cost_per_record_usd"] = round(
                led.get("confirmed_usd", 0) / v["delivered_records"], 4)
    return v


# Figure -> (regexes that would appear in a DM). Only load-bearing claims; page
# numbers, entry ids and dates are deliberately not tracked.
CLAIMS = {
    "delivered_records":        [r"5,226", r"5,228"],
    "total_pages":              [r"62,320"],
    "nonempty_transcriptions":  [r"62,209"],
    "failed_marker_pages":      [r"1,281"],
    "failed_marker_volumes":    [r"\b184\b"],
    "transcribed_volumes":      [r"\b232\b"],
    "hand_entries":             [r"3,452"],
    "hand_volumes":             [r"\bnine\b", r"\b9 volumes\b"],
    "gold_pages":               [r"\b335\b"],
    "gold_human_chars":         [r"421,756"],
    "pairs_scored":             [r"7\.3 million", r"7,305,667"],
    "pair_strata":              [r"\b444\b"],
    "pair_singletons":          [r"\b37\b"],
}


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve"}


def _states_value(text, value):
    """Does the DM state this figure, in any form a person would write it?

    Accepts the exact number, the comma-grouped form, the spelled-out word for
    small counts ("nine volumes"), and an explicit rounded form for large ones
    ("7.3 million" for 7,305,667).

    Without these the checker cries wolf on correct prose, and a checker that
    cries wolf gets ignored -- which would defeat the point of having one. It
    flagged four figures on its first run and three were prose it should have
    accepted.
    """
    if value is None:
        return False
    if isinstance(value, float):
        return (f"{value:.3f}" in text or f"{value:.2f}" in text
                or str(value) in text)
    if not isinstance(value, int):
        return str(value) in text
    if f"{value:,}" in text or str(value) in text:
        return True
    word = _WORDS.get(value)
    if word and re.search(r"\b" + word + r"\b", text, re.I):
        return True
    if value >= 1_000_000:                      # "7.3 million"
        m = f"{round(value / 1_000_000, 1)}"
        if re.search(re.escape(m) + r"\s*million", text, re.I):
            return True
    if value >= 1000:                           # "62.3k" / "~62,000"
        for form in (f"{value / 1000:.1f}k", f"{round(value, -3):,}"):
            if form in text:
                return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--dms", default="production/luna_v3")
    ap.add_argument("--transcriptions", default="../transcriptions/json")
    ap.add_argument("--manual", default="../ssda-openai/json")
    args = ap.parse_args(argv)

    v = recompute(args.root, args.transcriptions, args.manual)
    print("RECOMPUTED FROM ARTIFACTS")
    for k in sorted(v):
        print(f"   {k:28s} {v[k]}")

    dms = sorted(glob.glob(os.path.join(args.dms, "DM_*.md")))
    print(f"\nCHECKING {len(dms)} DM drafts")
    problems = []
    for p in dms:
        text = open(p, encoding="utf-8").read()
        for key, pats in CLAIMS.items():
            if key not in v or v[key] is None:
                continue
            hit = any(re.search(pat, text, re.I) for pat in pats)
            if not hit:
                continue
            # the DM mentions this quantity -- does the current value appear?
            if _states_value(text, v[key]):
                continue
            problems.append((os.path.basename(p), key, v[key],
                             [pat for pat in pats if re.search(pat, text, re.I)]))

    if not problems:
        print("   every tracked figure in every DM matches the current artifacts")
    else:
        print(f"   {len(problems)} figure(s) no longer match:")
        for f, key, now, seen in problems:
            print(f"     {f}: {key} is now {now:,}; DM still says {seen}")
        print("\n   STALE is not the same as WRONG. Check whether the corpus moved")
        print("   under the claim before assuming the DM was ever incorrect.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
