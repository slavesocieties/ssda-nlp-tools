#!/usr/bin/env python3
"""vocab_ab_report.py — did the vocabulary-aware prompt actually work?

Offline ($0, no network, no key). Compares two supplied 701054 extractions
per field and prints a verdict. It is usable both for the original vocabulary
experiment and for a completed corpus re-extraction.

    python vocab_ab_report.py

Baseline (old prompt, measured 2026-07-24): age 26.6%, ethnicity 0.0%,
phenotype 29.3%, relationship_type 98.9%.

Read the phenotype row with care: on 701054 every non-conformant phenotype value
is a correct Portuguese term missing from vocab.json (`preto`/`preta`) or a
feminine form of a listed masculine entry (`parda`/`branca`). No prompt can fix
that — it is a vocabulary gap, pending Daniel's call. Ethnicity is also a
historically open-ended field: its vocabulary rate is a drift diagnostic, not a
quality verdict. Judge this prompt on **age**; review ethnicity term-by-term.
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
# Age is a closed extraction category. Ethnicity and phenotype contain
# historically meaningful terms absent from the representative source vocab, so
# they must be audited rather than treated as prompt regressions.
JUDGE_ON = ["age"]
OPEN_DIAGNOSTICS = ["ethnicity", "phenotype"]


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
        star = ("  <-- judged" if f in JUDGE_ON else
                "  (open-vocabulary diagnostic; audit terms)" if f in OPEN_DIAGNOSTICS else "")
        print(f"{f:18s} {o_s:>18s} {n_s:>18s} {d_s:>9s}{star}")
        if o is not None and n is not None:
            verdict[f] = n - o

    print("\nVERDICT (age only; ethnicity/phenotype require term-level audit):")
    # A dimension with no values in the NEW extraction is NOT evidence of
    # improvement: the prompt tells the model to omit unsupported fields, so an
    # empty field is simply unmeasured. Say so rather than counting it as a win.
    measured = {f: verdict[f] for f in JUDGE_ON if f in verdict}
    unmeasured = [f for f in JUDGE_ON if f not in verdict]
    for f in unmeasured:
        n_had = new_t[f]
        print(f"  {f}: UNMEASURED — {'the new extraction emitted no values' if not n_had else 'no baseline to compare'}"
              f" (cannot count as improvement)")
    improved = [f for f, d in measured.items() if d > 5]
    regressed = [f for f, d in measured.items() if d < -5]
    if not measured:
        print("  INCONCLUSIVE — neither judged dimension had comparable values.")
        print("  -> do NOT spend on a full re-extraction on this evidence.")
    elif regressed:
        print(f"  REGRESSION in {', '.join(regressed)}. Do NOT re-extract; diagnose first.")
    elif len(improved) == len(measured) and len(measured) == len(JUDGE_ON):
        print(f"  PROMPT WORKS on the closed judged dimension ({', '.join(improved)}).")
        print("  -> do not re-extract solely to chase open-vocabulary conformance; audit terms first.")
    elif improved:
        print(f"  PARTIAL — {', '.join(improved)} improved, but "
              f"{len(JUDGE_ON) - len(improved)} of {len(JUDGE_ON)} judged dimension(s) "
              f"{'regressed or' if regressed else 'were flat or'} unmeasured.")
        print("  -> weigh the off-vocab values below before committing ~$15.")
    else:
        print("  NO IMPROVEMENT on the closed judged dimension. Do NOT spend on a re-extraction; diagnose first.")

    if new_off:
        print("\nremaining off-vocabulary values in the NEW extraction:")
        for f, c in sorted(new_off.items()):
            print(f"  {f:18s} {dict(c.most_common(6))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
