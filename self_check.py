#!/usr/bin/env python3
"""self_check.py — the failure modes that actually produced wrong claims here.

Offline, $0, no network, no key.

    python self_check.py

This is not a style guide. Every rule below is derived from a specific wrong
statement made in this repository, and each one is checked mechanically where
that is possible. Run it before reporting numbers.

THE ONE PATTERN BEHIND ALMOST ALL OF THEM
-----------------------------------------
A measurement returned a clean, flattering answer and I reported it. In every
case the harness was broken in a way that produced exactly the answer I wanted:

  * `pair_score` on label pointers returned 0.00 for every pair, because the
    pointers carry no "name". Against a 0.86 threshold that reads as "refused",
    so it reported ALL TEN of Daniel's negatives as correctly fixed.
  * `--no-lifespan` set an argparse value nothing consumed, so the A/B control
    had the guard ON in both arms and showed a delta of exactly zero on four
    metrics -- which reads as "the change does nothing".
  * A water-fill test built ONE stratum and asserted the depth spread was 0.
  * A mutation test's replacement string did not match, so it "passed" against
    unmodified code twice, appearing to prove a security test could fail.

The tell is always the same: THE RESULT IS TOO CLEAN. 10/10, exactly zero,
spread 0, 100%. Before believing a tidy number, make the harness produce a
result you know is WRONG. If it cannot fail, it is not measuring.

THE RULES
---------
1.  CONTROL FIRST. Prove the instrument moves before trusting what it reads.
    A control that shows no difference is indistinguishable from a control that
    was never wired up.
2.  SCORE IS NOT DISPOSITION. A pair can score 1.00 and be refused by a later
    guard. Read outcomes off the delivered artifact, never re-derive them.
    (Reading scores said 8 of 10 negatives "still merge"; the pipeline merged 1.)
3.  PAIRS ARE NOT PEOPLE. ISSUES ARE NOT RECORDS. 5,824 pair comparisons were
    534 name pairs over 1,982 mentions. "363 records" was 244 records and 363
    issues. Always report the unit the decision is made in.
4.  A BLOCK COUNT IS NOT AN IMPACT. The chronology guard blocked 1,416 merges;
    1,305 were already blocked by another rule. Attributable effect: 33 people.
5.  COMPARE LIKE WITH LIKE. v7 vs v8 moved 22,801 -> 32,943 identities because
    the corpus grew underneath it. `corpus_final_pipeline/` predates the current
    merge guards, so any diff against it is confounded.
6.  SCOPE YOUR SCAN. "Label ids match only 200 of 1,314" came from comparing the
    whole label set against ONE volume's ids. They match 1,314 of 1,314.
7.  IDS ARE LOCAL. Person ids (P01, P02) repeat in every entry. "P08 exists in
    the next entry" means only that the next entry has eight people.
8.  NAME THE UNVERIFIED. 375062's text looks column-interleaved; the proxy for
    it does not track the defect rate, so that stays a hypothesis, not a cause.
9.  NEVER TRUNCATE SILENTLY. A page capped at 2,000 of 5,824 rows reads as
    complete. Say what was withheld, in the output.
10. THE TIDIER FIX IS OFTEN CATASTROPHIC. Unioning both edge directions before
    testing contradictions turned 5 real pairs into 10,050, because A->B parent
    plus B->A child is every real parent.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("gold volumes overlap the delivered corpus")
def _gold_overlap(root):
    """Rule 5. Accuracy measured on volumes we do not ship bounds the pipeline,
    not the data. Currently the overlap is ZERO, which every statement of
    'substitution 6.31%' omitted."""
    g = os.path.join(root, "production/luna_v3/manual_gold.json")
    if not os.path.exists(g):
        return None, "no manual_gold.json"
    gold = set((json.load(open(g, encoding="utf-8")).get("volumes") or {}))
    deliv = {os.path.basename(p).split(".")[0] for p in
             glob.glob(os.path.join(root, "production/luna_v3/assembled/*.materialized.json"))}
    both = gold & deliv
    return (bool(both),
            f"gold {sorted(gold)} vs delivered {len(deliv)} volumes; overlap "
            f"{sorted(both) or 'NONE -- quote these figures as pipeline evidence only'}")


@check("no A/B run reports a zero delta on every metric")
def _ab_not_null(root):
    """Rule 1. Exactly-zero everywhere is the signature of a control that was
    never wired up, not of a change that does nothing."""
    stats = sorted(glob.glob(os.path.join(root, "production/luna_v3/merge/*.stats.json")))
    if len(stats) < 2:
        return None, "fewer than two merge runs to compare"
    keys = ("identities", "merged_identities", "auto_merges")
    runs = {os.path.basename(p): json.load(open(p, encoding="utf-8")) for p in stats}
    suspicious = []
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = runs[names[i]], runs[names[j]]
            if a.get("mentions") != b.get("mentions"):
                continue                       # different corpora, not an A/B
            if all(a.get(k) == b.get(k) for k in keys):
                suspicious.append(f"{names[i]} == {names[j]}")
    return (not suspicious,
            "identical on every metric: " + "; ".join(suspicious) if suspicious
            else f"{len(stats)} runs, no all-zero deltas")


@check("every capped output declares what it withheld")
def _no_silent_cap(root):
    """Rule 9. A tool that slices [:N] must say so in its own output."""
    # Slicing a LIST (`][:n]`) drops data. Slicing a function result (`)[:n]`)
    # is nearly always `str(e)[:110]` or `json.dumps(row)[:120]` shortening one
    # line for display. A first version conflated them and named 11 files, 3 of
    # the 4 inspected being display truncation -- a checker that cries wolf gets
    # ignored, which is the same failure as one that stays silent.
    offenders = []
    for f in sorted(glob.glob(os.path.join(root, "*.py"))):
        src = open(f, encoding="utf-8", errors="ignore").read()
        if re.search(r"\]\[:\s*(?:args\.)?(?:limit|top|max|n|\d{2,})\s*\]", src):
            says = re.search(r"NOT shown|withheld|capped|not persisted|no top-N",
                             src, re.I)
            if not says:
                offenders.append(os.path.basename(f))
    return not offenders, ("silent truncation in: " + ", ".join(offenders)
                           if offenders else "all capped outputs declare the cap")


@check("tests exist for the traps that produced wrong numbers")
def _trap_tests(root):
    """Each of these pins a specific wrong result. Losing one loses the lesson."""
    want = {
        "tests/test_compare_merge_runs.py": "control that is not a control",
        "tests/test_verify_label_scores.py": "degenerate scoring harness",
        "tests/test_validate_training_sample.py": "reservoir that cannot fail",
        "tests/test_validate_graph.py": "ordered-pair double count",
        "tests/test_security.py": "script breakout in review pages",
    }
    missing = [p for p in want if not os.path.exists(os.path.join(root, p))]
    return not missing, ("missing: " + ", ".join(missing) if missing
                         else f"all {len(want)} trap suites present")


@check("the test suite passes")
def _suite(root):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                        os.path.join(root, "tests")],
                       capture_output=True, text=True, cwd=root)
    last = [l for l in r.stdout.strip().split("\n") if l.strip()][-1:]
    return r.returncode == 0, (last[0] if last else "no output")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    print("SELF-CHECK -- rules derived from actual wrong claims in this repo\n")
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(args.root)
        except Exception as e:                      # a check must never mask itself
            ok, detail = False, f"check raised {type(e).__name__}: {e}"
        mark = "n/a " if ok is None else ("OK  " if ok else "FAIL")
        failed += (ok is False)
        print(f"  [{mark}] {name}")
        print(f"         {detail}")
    print(f"\n{failed} failing check(s).")
    print("\nThe rules this cannot check mechanically are in the module docstring."
          "\nThe important one: if a result is suspiciously clean, break the "
          "harness\non purpose and confirm it can produce a wrong answer.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
