#!/usr/bin/env python3
"""vocab_ab_report.py — did the vocabulary-aware prompt actually work?

Offline ($0, no network, no key). Compares controlled-vocabulary conformance
between the OLD extraction of 701054 and the NEW `701054-vocabtest` extraction,
per field, and prints a verdict.

    python vocab_ab_report.py

Baseline (old prompt, measured 2026-07-24): age 26.6%, ethnicity 0.0%,
phenotype 29.3%, relationship_type 98.9%.

Read the phenotype row with care: on 701054 every non-conformant phenotype value
is a correct Portuguese term missing from vocab.json (`preto`/`preta`) or a
feminine form of a listed masculine entry (`parda`/`branca`). No prompt can fix
that — it is a vocabulary gap, pending Daniel's call. Judge the prompt on **age**
and **ethnicity**.
"""
import argparse
import glob
import json
import os
from collections import Counter

from ssda_nlp_tools import vocab as V

FIELDS = ["age", "ethnicity", "phenotype", "occupation", "rank"]
BASELINE = {"age": 26.6, "ethnicity": 0.0, "phenotype": 29.3,
            "relationship_type": 98.9, "occupation": 99.4, "rank": 100.0}
# Prompt-fixable dimensions. phenotype is excluded on purpose (vocab gap).
JUDGE_ON = ["age", "ethnicity"]


def conformance(records, v):
    tot, ok, off = Counter(), Counter(), {}
    for r in records:
        for p in (r.get("data") or {}).get("people") or []:
            for f in FIELDS:
                val = p.get(f)
                if val in (None, "", []):
                    continue
                tot[f] += 1
                if v.is_known(f, val):
                    ok[f] += 1
                else:
                    off.setdefault(f, Counter())[str(val)] += 1
            for rel in p.get("relationships") or []:
                val = rel.get("relationship_type")
                if not val:
                    continue
                tot["relationship_type"] += 1
                if v.is_known("relationship_type", val):
                    ok["relationship_type"] += 1
                else:
                    off.setdefault("relationship_type", Counter())[str(val)] += 1
    return tot, ok, off


def load_records(paths):
    out = []
    for p in paths:
        d = json.loads(open(p, encoding="utf-8").read())
        out.extend(d.get("entries") or d.get("records") or [])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="production/luna_live/assembled/701054-vocabtest.materialized.json",
                    help="the vocabulary-aware extraction")
    ap.add_argument("--old", default="production/luna_live/701054.materialized.json",
                    help="the pre-fix extraction (baseline)")
    args = ap.parse_args(argv)

    v = V.load_vocab()
    if not os.path.exists(args.new):
        alt = sorted(glob.glob("production/luna_live/**/*vocabtest*.json", recursive=True))
        print(f"NEW extraction not found at {args.new}")
        print("  candidates on disk:", alt or "(none yet — the batch has not been assembled)")
        print("  run: python assemble_corpus.py   (after the monitor validates the job)")
        return 1

    old_t, old_o, _ = conformance(load_records([args.old]), v)
    new_t, new_o, new_off = conformance(load_records([args.new]), v)

    print("701054 controlled-vocabulary conformance — old prompt vs vocabulary-aware prompt\n")
    print(f"{'field':18s} {'old':>18s} {'new':>18s} {'delta':>9s}")
    verdict = {}
    for f in ["age", "ethnicity", "phenotype", "relationship_type", "occupation", "rank"]:
        if not new_t[f] and not old_t[f]:
            continue
        o = 100 * old_o[f] / old_t[f] if old_t[f] else None
        n = 100 * new_o[f] / new_t[f] if new_t[f] else None
        o_s = f"{old_o[f]:4d}/{old_t[f]:<4d} {o:5.1f}%" if o is not None else f"{'—':>15s}"
        n_s = f"{new_o[f]:4d}/{new_t[f]:<4d} {n:5.1f}%" if n is not None else f"{'—':>15s}"
        d_s = f"{n-o:+8.1f}" if (o is not None and n is not None) else "       —"
        star = "  <-- judged" if f in JUDGE_ON else ("  (vocab gap)" if f == "phenotype" else "")
        print(f"{f:18s} {o_s:>18s} {n_s:>18s} {d_s:>9s}{star}")
        if o is not None and n is not None:
            verdict[f] = n - o

    print("\nVERDICT (age + ethnicity only — phenotype is a vocabulary gap, not a prompt issue):")
    judged = [verdict[f] for f in JUDGE_ON if f in verdict]
    if not judged:
        print("  inconclusive — no comparable values")
    elif all(d > 5 for d in judged):
        print("  PROMPT WORKS. Both judged dimensions improved materially.")
        print("  -> re-extracting the remaining volumes is justified (~$15 Batch API).")
    elif any(d > 5 for d in judged):
        print("  PARTIAL. One dimension improved, the other did not — inspect the off-vocab")
        print("  values below before committing to a full re-run.")
    else:
        print("  NO IMPROVEMENT. Do NOT spend ~$15 on a full re-extraction; diagnose first.")

    if new_off:
        print("\nremaining off-vocabulary values in the NEW extraction:")
        for f, c in sorted(new_off.items()):
            print(f"  {f:18s} {dict(c.most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
