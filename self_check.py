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


@check("no merge artifact is empty or truncated")
def _artifacts_intact(root):
    """The class of defect the 2026-08-05 review found, and this file MISSED.

    Four tools accepted an empty corpus, wrote `mentions: 0`, and overwrote real
    results with it. self_check's A/B rule then SKIPPED the corrupted run, because
    it only compares runs of equal mention count and nothing else had zero. A
    checker that ignores a destroyed artifact is worse than no checker: the run
    is still listed, still loadable, and still wrong.
    """
    import glob as _g
    bad = []
    for p in sorted(_g.glob(os.path.join(root, "production/luna_v3/merge/*.stats.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            bad.append(f"{os.path.basename(p)} unreadable ({type(e).__name__})")
            continue
        if not d.get("mentions"):
            bad.append(f"{os.path.basename(p)} has mentions={d.get('mentions')!r}")
    return (not bad,
            "empty/corrupt: " + "; ".join(bad) if bad
            else "every merge run records a non-zero mention count")


@check("artifact-writing tools refuse an empty corpus (EXECUTED, not inspected)")
def _tools_refuse_empty(root):
    """THE GAP THIS FILE HAD. Every other check here examines FINDINGS -- the
    numbers, the artifacts, the tests. None examined the TOOLS, which is why the
    empty-corpus bug went unnoticed: four scripts wrote `mentions: 0` over real
    results, exited zero, and nothing in this file looked.

    So this one RUNS them. Each artifact-writing tool is invoked against an empty
    directory in a subprocess, and must write nothing and fail with a message
    rather than a traceback. A grep for `require_corpus` would pass a tool whose
    guard is unreachable; executing it cannot be fooled that way.
    """
    import subprocess as sp
    import tempfile

    tools = sorted(
        os.path.basename(f) for f in glob.glob(os.path.join(root, "*.py"))
        if "--assembled" in open(f, encoding="utf-8", errors="ignore").read()
        and re.search(r"json\.dump|open\([^)]*[\"']w[\"']",
                      open(f, encoding="utf-8", errors="ignore").read()))
    if not tools:
        return None, "no artifact-writing tools found"

    empty = tempfile.mkdtemp()
    bad = []
    for t in tools:
        before = _snapshot(root)
        r = sp.run([sys.executable, "-X", "utf8", t, "--assembled", empty],
                   capture_output=True, text=True, cwd=root, timeout=120)
        wrote = _snapshot(root) - before
        if wrote:
            bad.append(f"{t} WROTE {len(wrote)} file(s) on empty input")
        elif "Traceback" in (r.stderr or ""):
            last = [l for l in r.stderr.strip().splitlines() if l.strip()][-1]
            bad.append(f"{t} crashed: {last[:48]}")
    return (not bad, "; ".join(bad) if bad
            else f"all {len(tools)} tools refuse empty input cleanly")


def _snapshot(root):
    """Mtimes of everything under production/, to detect any write."""
    out = set()
    for dp, _, fs in os.walk(os.path.join(root, "production")):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                out.add((p, os.path.getmtime(p)))
            except OSError:
                pass
    return out


# Calls that actually reach the network or a paid API, matched against CODE with
# comments stripped. A docstring, or a default path like "../ssda-openai/...",
# is not an API call -- a cruder pattern flagged run_evidence_merge.py and
# verify_claims.py, both offline tools written the same day, and shrinking the
# executed set is the failure mode here: an unexecuted tool is an unchecked one.
_SIDE_EFFECTING = re.compile(
    r"urlopen\(|urllib\.request\.(?:urlopen|Request)|requests\.(?:get|post|put)"
    r"|\.create\(|boto3\.|client\.(?:messages|chat|completions)"
    r"|environ(?:\.get\(|\[)\s*[\"'][A-Z_]*KEY")


# Scripts that have NEVER imported in this repository, with the reason. An
# acknowledged failure is not a tolerated one: a checker carrying a permanent
# red line gets ignored, which is the cry-wolf failure that made the
# truncation rule useless at 11 hits. Anything NOT on this list breaking is a
# regression and fails the check.
KNOWN_BROKEN = {
    "transcription_json_to_training_rule_based.py":
        "inherited from the original SSDA repo; imports `eval_entry`, which "
        "does not exist here and never has. Nothing calls it -- the only "
        "reference is eval_data/zekai_algorithm_review.md. Fixing it needs the "
        "missing module, not a guess at what check_entry did.",
}


def _is_offline(path):
    src = open(path, encoding="utf-8", errors="ignore").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    return not _SIDE_EFFECTING.search(code)


@check("every offline script imports and answers --help (EXECUTED)")
def _all_scripts_import(root):
    """Widened from the 12 artifact-writers to the whole repository.

    `--help` is the strongest thing that is unconditionally SAFE to run: it
    proves the module imports, that argparse is well formed, and that nothing
    explodes at import time -- while touching no data and calling nothing.

    Scripts that reach the network or a paid API are NEVER executed, whatever
    they would do. That exclusion is deliberate and not a coverage compromise:
    running them could spend money or hit a rate limit, and no check is worth
    that.
    """
    import subprocess as sp
    every = [f for f in sorted(glob.glob(os.path.join(root, "*.py")))
             if os.path.basename(f) != "self_check.py"]
    scripts = [f for f in every if _is_offline(f)]
    skipped = len(every) - len(scripts)
    bad, known = [], []
    for f in scripts:
        name = os.path.basename(f)
        r = sp.run([sys.executable, "-X", "utf8", name, "--help"],
                   capture_output=True, text=True, cwd=root, timeout=120)
        if "Traceback" not in (r.stderr or ""):
            continue
        last = [l for l in r.stderr.strip().splitlines() if l.strip()][-1]
        (known if name in KNOWN_BROKEN else bad).append(f"{name}: {last[:50]}")
    note = f" ({len(known)} known-broken acknowledged)" if known else ""
    shown = "; ".join(bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else "")
    return (not bad, shown if bad else
            f"{len(scripts)} offline scripts import cleanly{note} "
            f"({skipped} network/paid never executed)")


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
