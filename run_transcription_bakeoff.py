#!/usr/bin/env python3
"""run_transcription_bakeoff.py — gemini-3.1-pro vs gpt-5.6-luna, for what
Archivault actually does: reading colonial handwriting off a page image.

Four subcommands, in the order you should run them. Only `probe` and `judge`
touch a network or a key. Both require --confirm; the judge also requires an
explicit local reservation and cap before it can send anything.

  probe    Does the Archivault backend even ACCEPT the candidate model string?
           Nothing in the repo validates it -- `transcription_model` is passed
           straight through -- and the API docs do not list supported models. So
           this is unanswerable except by trying it on one page. Cheap, and it
           gates everything else.

  segment  Offline, $0. Convert the page-level Archivault output from each
           model into the deterministic, cross-page segmented JSON that score
           consumes. It is deliberately explicit so raw pages can never be
           mistaken for an evaluated transcription.

  score    Offline, $0. Runs our segmenter's metrics over two already-produced
           transcriptions and reports which better supports the downstream work.
           See ssda_nlp_tools/transcription_bakeoff for why these metrics and
           not character error rate.

  judge    An explicitly selected Anthropic vision model adjudicates the pages
           where the two transcriptions diverge MOST. It is a guarded, opt-in
           SCREEN rather than a verdict -- see the warning under `judge`.

  report   Builds a side-by-side HTML of the divergent passages for Daniel,
           which is the only route to an actual accuracy judgement.

Credentials are read from the environment and never written, printed, or passed
on a command line.
"""
import argparse
import importlib.util
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

from ssda_nlp_tools.transcription_bakeoff import (compare, divergent_pages,
                                                  score_transcription)

BASELINE = "gemini-3.1-pro-preview"      # what every production script hardcodes
CANDIDATE = "gpt-5.6-luna"


def cmd_probe(args):
    """One page, one model, through Archivault without exposing its password."""
    submit = Path(args.archivault) / "submit_job.py"
    if not submit.exists():
        sys.exit(f"submit_job.py not found at {submit}. Clone "
                 "https://github.com/slavesocieties/ssda-archivault first.")
    source_argv = []
    if args.keys_file:
        key_path = Path(args.keys_file)
        if not key_path.exists():
            sys.exit(f"S3 key file not found: {key_path}")
        keys = [line.strip() for line in key_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(keys) != 1:
            sys.exit("the probe requires exactly one non-empty S3 key in --keys-file")
        source_argv = ["--source-bucket", args.bucket, "--keys", keys[0]]
    else:
        image = Path(args.local_image)
        if not image.is_file():
            sys.exit(f"local probe image not found: {image}")
        # Archivault's local-upload mode accepts a directory. Copy exactly one
        # immutable source image into an auditable staging directory so no
        # sibling pages can silently widen this one-page paid probe.
        staging = Path(args.outdir) / "source_image"
        staging.mkdir(parents=True, exist_ok=True)
        staged_image = staging / image.name
        shutil.copy2(image, staged_image)
        source_argv = ["--dir", str(staging)]
    argv = [str(submit),
           "--email", args.email,
           "--title", f"model-probe-{args.model}",
           "--steps", "transcribe",
           "--transcription-model", args.model,
           *source_argv,
           "--out-dir", args.outdir,
           "--language", args.language, "--writing-style", "handwritten",
           "--time-period", "19th_century_or_earlier"]
    shown = list(argv)
    if "--keys" in shown:
        shown[shown.index("--keys") + 1] = "[one S3 key]"
    else:
        shown[shown.index("--dir") + 1] = "[one staged local image]"
    print("  " + " ".join(shown) + " --password [from ARCHIVAULT_PASSWORD]")
    if not args.confirm:
        print("\nDRY RUN -- rerun with --confirm to submit. One page only.")
        return 0
    if not os.environ.get("ARCHIVAULT_PASSWORD"):
        sys.exit("set ARCHIVAULT_PASSWORD in the environment first; this script "
                 "never takes a password on the command line.")
    if importlib.util.find_spec("requests") is None:
        sys.exit("the selected Python interpreter lacks Archivault's `requests` "
                 "dependency; use an interpreter with requests installed. No "
                 "network call was made.")
    # Archivault's CLI requires --password.  Run it in this process and only
    # append the secret to its *in-memory* argv after process start: neither a
    # shell history nor an OS process listing can contain it.
    old_argv = sys.argv
    try:
        sys.argv = argv + ["--password", os.environ["ARCHIVAULT_PASSWORD"]]
        runpy.run_path(str(submit), run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
    if code:
        print(f"\nsubmit_job.py exited {code}. If the backend rejected "
              f"'{args.model}', that is the answer: the model is not available "
              f"through Archivault and the comparison stops here.")
    return code


def cmd_score(args):
    a = json.load(open(args.baseline, encoding="utf-8"))
    b = json.load(open(args.candidate, encoding="utf-8"))
    sa = score_transcription(a)
    sb = score_transcription(b)
    if not sa["entries"] or not sb["entries"]:
        sys.exit("refusing to score an output with zero segmented entries; check "
                 "the Archivault artifact and run `segment` on a real register page")
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


def cmd_segment(args):
    """Run the project's tested deterministic segmenter over one model output."""
    from run_segment import main as segment_main
    argv = [args.input, "--out", args.out]
    if args.structural:
        argv.append("--structural")
    return segment_main(argv)


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
    if args.reservation_per_page <= 0 or args.max_usd <= 0:
        sys.exit("--reservation-per-page and --max-usd must both be positive")
    div = json.load(open(args.divergent, encoding="utf-8"))[:args.limit]
    imgs = Path(args.images)
    missing = [d["image"] for d in div if not (imgs / d["image"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} page image(s) not found under {imgs}, "
                 f"e.g. {missing[0]}. The judge is worthless without the image.")
    reservation = len(div) * args.reservation_per_page
    print(f"{len(div)} pages, local reservation ${reservation:.2f} at "
          f"${args.reservation_per_page}/page (cap ${args.max_usd:.2f})")
    if reservation > args.max_usd + 1e-9:
        print("REFUSING: the declared local reservation exceeds --max-usd.")
        return 2
    if not args.confirm:
        print("DRY RUN -- no key access. Rerun with --confirm to call the API.")
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY in the environment first.")

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
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--keys-file", help="ONE S3 key, one line")
    source.add_argument("--local-image", help="one verified local source image")
    p.add_argument("--language", default="spanish")
    p.add_argument("--outdir", default="production/bakeoff/probe")
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_probe)

    g = sub.add_parser("segment", help="segment one raw Archivault transcription JSON offline")
    g.add_argument("input", help="page-level Archivault transcription JSON")
    g.add_argument("--out", required=True, help="segmented JSON output path")
    g.add_argument("--structural", action="store_true",
                   help="also compare segment starts with visible margin numbers")
    g.set_defaults(func=cmd_segment)

    s = sub.add_parser("score", help="offline downstream-usability comparison")
    s.add_argument("baseline"); s.add_argument("candidate")
    s.add_argument("--label-a", default="gemini-3.1-pro")
    s.add_argument("--label-b", default="luna")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--outdir", default="production/bakeoff")
    s.set_defaults(func=cmd_score)

    j = sub.add_parser("judge", help="Anthropic vision-model screen on divergent pages")
    j.add_argument("divergent", help="bakeoff_divergent.json from `score`")
    j.add_argument("--images", required=True, help="directory of page images")
    j.add_argument("--limit", type=int, default=15)
    j.add_argument("--reservation-per-page", type=float, required=True,
                   help="conservative local USD reservation per page")
    j.add_argument("--max-usd", type=float, required=True,
                   help="hard cap for this local reservation")
    j.add_argument("--model", required=True,
                   help="Anthropic model ID for the judge (not a contestant), "
                        "e.g. claude-opus-5. Required rather than defaulted: this "
                        "is a paid call, and the operator should name a model "
                        "their account actually has.")
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
