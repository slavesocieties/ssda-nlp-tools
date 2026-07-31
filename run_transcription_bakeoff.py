#!/usr/bin/env python3
"""run_transcription_bakeoff.py — gemini-3.1-pro vs gpt-5.6-luna, for what
Archivault actually does: reading colonial handwriting off a page image.

Four subcommands, in the order you should run them. Only `probe` and `judge`
touch a network or a key, and both are capped and require --confirm.

  probe    Does the Archivault backend even ACCEPT the candidate model string?
           Nothing in the repo validates it -- `transcription_model` is passed
           straight through -- and the API docs do not list supported models. So
           this is unanswerable except by trying it on one page. Cheap, and it
           gates everything else.

  score    Offline, $0. Runs our segmenter's metrics over two already-produced
           transcriptions and reports which better supports the downstream work.
           See ssda_nlp_tools/transcription_bakeoff for why these metrics and
           not character error rate.

  judge    Opus 5 adjudicates the pages where the two transcriptions diverge
           MOST, with the page image in front of it. Capped, opt-in, and a
           SCREEN rather than a verdict -- see the warning under `judge`.

  report   Builds a side-by-side HTML of the divergent passages for Daniel,
           which is the only route to an actual accuracy judgement.

Credentials are read from the environment and never written, printed, or passed
on a command line.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ssda_nlp_tools.transcription_bakeoff import (compare, divergent_pages,
                                                  score_transcription)

BASELINE = "gemini-3.1-pro-preview"      # what every production script hardcodes
CANDIDATE = "gpt-5.6-luna"


def cmd_probe(args):
    """One page, one model, straight through submit_job.py."""
    submit = Path(args.archivault) / "submit_job.py"
    if not submit.exists():
        sys.exit(f"submit_job.py not found at {submit}. Clone "
                 "https://github.com/slavesocieties/ssda-archivault first.")
    if not os.environ.get("ARCHIVAULT_PASSWORD"):
        sys.exit("set ARCHIVAULT_PASSWORD in the environment first; this script "
                 "never takes a password on the command line.")
    cmd = [sys.executable, str(submit),
           "--source-bucket", args.bucket, "--email", args.email,
           "--title", f"model-probe-{args.model}",
           "--steps", "transcribe",
           "--transcription-model", args.model,
           "--keys-file", args.keys_file,
           "--out-dir", args.outdir,
           "--language", args.language, "--writing-style", "handwritten",
           "--time-period", "19th_century_or_earlier"]
    print("  " + " ".join(cmd))
    if not args.confirm:
        print("\nDRY RUN -- rerun with --confirm to submit. One page only.")
        return 0
    # password travels in the child's environment, not argv, so it cannot leak
    # into a process listing or a shell history
    env = dict(os.environ)
    r = subprocess.run(cmd, env=env)
    if r.returncode:
        print(f"\nsubmit_job.py exited {r.returncode}. If the backend rejected "
              f"'{args.model}', that is the answer: the model is not available "
              f"through Archivault and the comparison stops here.")
    return r.returncode


def cmd_score(args):
    a = json.load(open(args.baseline, encoding="utf-8"))
    b = json.load(open(args.candidate, encoding="utf-8"))
    sa = score_transcription(a)
    sb = score_transcription(b)
    res = compare(sa, sb, args.label_a, args.label_b)

    w = max(len(r["metric"]) for r in res["rows"]) + 2
    print(f"{'metric':{w}s} {args.label_a:>20s} {args.label_b:>20s} "
          f"{'diff':>8s}   better")
    for r in res["rows"]:
        print(f"  {r['metric']:{w-2}s} {str(r[args.label_a]):>20s} "
              f"{str(r[args.label_b]):>20s} {100*r['rel_diff']:7.1f}%   {r['better']}")
    print(f"\n  {res['wins']}")
    print(f"  VERDICT: {res['verdict']}")

    div = divergent_pages(a, b, top=args.top)
    if div:
        print(f"\n  {len(div)} most-divergent pages (similarity {div[0]['similarity']}"
              f" to {div[-1]['similarity']})")
        for d in div[:5]:
            print(f"    {d['image']}  {d['similarity']}")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "bakeoff_score.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": sa, "candidate": sb, "comparison": res,
                   "divergent": [{k: v for k, v in d.items() if k != "a" and k != "b"}
                                 for d in div]}, f, ensure_ascii=False, indent=2)
    with open(out / "bakeoff_divergent.json", "w", encoding="utf-8") as f:
        json.dump(div, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out/'bakeoff_score.json'}\n-> {out/'bakeoff_divergent.json'}")
    return 0


JUDGE_PROMPT = """You are shown one page of a colonial sacramental register and two
independent transcriptions of it, A and B.

Judge ONLY which transcription is more faithful to the handwriting in the image.
Do not reward fluency, modernised spelling, or expanded abbreviations: a faithful
transcription of a damaged page may be ugly and full of gaps.

Weigh, in order:
1. Words visibly present in the image but missing from a transcription.
2. Words present in a transcription that you cannot find in the image.
3. Names, dates and numbers, which matter more than connective prose.
4. Line and entry boundaries.

If the handwriting is too degraded for you to tell, say so. "unclear" is a more
useful answer than a guess, and you will not be penalised for it.

Reply as JSON only:
{"winner": "A" | "B" | "tie" | "unclear",
 "confidence": 0.0-1.0,
 "evidence": ["short specific observation", ...]}"""


def cmd_judge(args):
    """Opus 5 with the image in front of it.

    WARNING, and it belongs in the output rather than a footnote: this judge is
    reading the same degraded handwriting the models are, and is not a
    palaeographer. It is a SCREEN -- good for ranking where to look and for
    catching a model that hallucinates whole clauses -- not a verdict. Where it
    reports low confidence or `unclear`, believe that. The accuracy question is
    settled by Daniel reading the pages this stage surfaces.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY in the environment first.")
    div = json.load(open(args.divergent, encoding="utf-8"))[:args.limit]
    imgs = Path(args.images)
    missing = [d["image"] for d in div if not (imgs / d["image"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} page image(s) not found under {imgs}, "
                 f"e.g. {missing[0]}. The judge is worthless without the image.")
    est = len(div) * args.cost_per_page
    print(f"{len(div)} pages, ~${est:.2f} at ${args.cost_per_page}/page "
          f"(cap {args.limit})")
    if not args.confirm:
        print("DRY RUN -- rerun with --confirm to call the API.")
        return 0

    import base64
    import urllib.request

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, d in enumerate(div, 1):
        img = (imgs / d["image"]).read_bytes()
        body = {
            "model": args.model,
            "max_tokens": 900,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg",
                                             "data": base64.b64encode(img).decode()}},
                {"type": "text", "text": JUDGE_PROMPT
                 + f"\n\n--- TRANSCRIPTION A ---\n{d['a'][:6000]}"
                   f"\n\n--- TRANSCRIPTION B ---\n{d['b'][:6000]}"},
            ]}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json",
                     "anthropic-version": "2023-06-01",
                     # read from the environment, never logged or echoed
                     "x-api-key": os.environ["ANTHROPIC_API_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read())
            text = "".join(c.get("text", "") for c in payload.get("content", []))
            try:
                verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError):
                verdict = {"winner": "unparsed", "confidence": 0.0, "raw": text[:400]}
        except Exception as exc:                       # one bad page must not
            verdict = {"winner": "error", "confidence": 0.0,   # lose the others
                       "error": f"{type(exc).__name__}: {exc}"}
        verdict["image"] = d["image"]
        verdict["similarity"] = d["similarity"]
        results.append(verdict)
        print(f"  [{i}/{len(div)}] {d['image']}  -> {verdict.get('winner')} "
              f"({verdict.get('confidence')})")

    tally = {}
    for r in results:
        tally[r.get("winner", "?")] = tally.get(r.get("winner", "?"), 0) + 1
    confident = [r for r in results
                 if r.get("winner") in ("A", "B") and (r.get("confidence") or 0) >= 0.6]
    with open(out / "judge_results.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "tally": tally, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  tally: {tally}")
    print(f"  confident (>=0.6) calls: {len(confident)} of {len(results)}")
    print("\n  This is a SCREEN, not a verdict. The judge is reading the same")
    print("  degraded hand the models are and is not a palaeographer; treat")
    print("  'unclear' and low confidence as real. Daniel reading the pages")
    print("  surfaced here is what settles accuracy.")
    print(f"\n-> {out/'judge_results.json'}")


def cmd_report(args):
    from ssda_nlp_tools.bakeoff_html import render_bakeoff_html
    div = json.load(open(args.divergent, encoding="utf-8"))[:args.limit]
    out = render_bakeoff_html(div, args.out, label_a=args.label_a,
                              label_b=args.label_b)
    print(f"{len(div)} divergent pages -> {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="does the backend accept the model?")
    p.add_argument("--archivault", default="../../ssda-archivault")
    p.add_argument("--model", default=CANDIDATE)
    p.add_argument("--email", required=True)
    p.add_argument("--bucket", default="ssda-production-jpgs")
    p.add_argument("--keys-file", required=True, help="ONE S3 key, one line")
    p.add_argument("--language", default="spanish")
    p.add_argument("--outdir", default="production/bakeoff/probe")
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_probe)

    s = sub.add_parser("score", help="offline downstream-usability comparison")
    s.add_argument("baseline"); s.add_argument("candidate")
    s.add_argument("--label-a", default="gemini-3.1-pro")
    s.add_argument("--label-b", default="luna")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--outdir", default="production/bakeoff")
    s.set_defaults(func=cmd_score)

    j = sub.add_parser("judge", help="Opus 5 screen on divergent pages")
    j.add_argument("divergent", help="bakeoff_divergent.json from `score`")
    j.add_argument("--images", required=True, help="directory of page images")
    j.add_argument("--limit", type=int, default=15)
    j.add_argument("--cost-per-page", type=float, default=0.05)
    j.add_argument("--model", default="claude-opus-5",
                   help="the judge, not a contestant")
    j.add_argument("--outdir", default="production/bakeoff")
    j.add_argument("--confirm", action="store_true")
    j.set_defaults(func=cmd_judge)

    r = sub.add_parser("report", help="side-by-side HTML for Daniel")
    r.add_argument("divergent")
    r.add_argument("--out", default="production/bakeoff/transcription_bakeoff.html")
    r.add_argument("--label-a", default="gemini-3.1-pro")
    r.add_argument("--label-b", default="luna")
    r.add_argument("--limit", type=int, default=20)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
