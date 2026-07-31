#!/usr/bin/env python3
"""run_probe_set.py — the four-page Luna probe, end to end, in one command.

    $env:ARCHIVAULT_PASSWORD = "..."
    python run_probe_set.py --email you@vanderbilt.edu --confirm

Only Luna is submitted. The gemini-3.1-pro side is lifted from the Archivault
transcription set we already hold, so the baseline is the ACTUAL production
output for that exact page rather than a fresh run that might differ from it.
That halves the paid calls and removes a confound.

Without --confirm this prints the plan and exits, having touched no network.
With --confirm, an explicit per-page reservation and persistent hard-cap ledger
are required; submitted upstream requests remain reserved until authoritative
billing evidence is reconciled.

Page selection (production/bakeoff/probe_set.json) spans the difficulty range
deliberately -- 1701 to 1907, Portuguese and Spanish, 450 to 2,900 characters of
baseline text. The first probe used a single 1841 Portuguese page; a second
sample of near-identical pages would tell us almost nothing new.

Read the verdict with the limitation in mind: the automated screen measures
whether text is WELL FORMED, and fluent fabrication is well formed. On the first
probe Luna won the vocabulary metric with invented text. The side-by-side HTML
and the images are what settle it.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ZIP = Path(r"C:/Users/mahajar/Downloads/sample images/"
           r"Archivault transcriptions-20260714T193242Z-1-001.zip")
IMAGES = Path(r"C:/Users/mahajar/Downloads/sample images/"
              r"_eccl_sample/ecclesiastical random sample")


def baseline_for(page: str) -> str:
    """The production gemini-3.1-pro transcription of exactly this page."""
    vol = page.split("-")[0]
    with zipfile.ZipFile(ZIP) as z:
        doc = json.loads(z.read(f"Archivault transcriptions/json/{vol}.json")
                         .decode("utf-8"))
    for rec in doc:
        if rec.get("file") == page:
            return rec.get("transcription") or ""
    raise KeyError(f"{page} not found in {vol}.json")


def as_volume(page: str, text: str) -> dict:
    """Wrap one page as the Archivault-shaped input `segment` consumes."""
    return {"volume": page.split("-")[0],
            "pages": [{"file": page, "transcription": text}]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-set", default="production/bakeoff/probe_set.json")
    ap.add_argument("--outdir", default="production/bakeoff/probe_set_run")
    ap.add_argument("--email", required=True)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--archivault", default=r"C:/Users/mahajar/Downloads/ssda-archivault")
    ap.add_argument("--reservation-usd", type=float,
                    help="conservative USD reservation per submitted Luna page")
    ap.add_argument("--max-usd", type=float,
                    help="hard cumulative USD cap for all pages in --ledger")
    ap.add_argument("--ledger", default="production/bakeoff/transcription_spend_ledger.json")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args(argv)

    pages = json.load(open(args.probe_set, encoding="utf-8"))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(pages)} pages, Luna only (gemini baseline reused from the "
          f"transcription set):\n")
    for p in pages:
        img = IMAGES / p["file"]
        print(f"  {p['file']:22s} {p['lang']}  {p['year']}  {p['label']:20s} "
              f"{'image OK' if img.is_file() else 'IMAGE MISSING'}")
    missing = [p["file"] for p in pages if not (IMAGES / p["file"]).is_file()]
    if missing:
        sys.exit(f"\n{len(missing)} image(s) missing, e.g. {missing[0]}")

    if not args.confirm:
        print(f"\nDRY RUN -- no network call. Rerun with --confirm.\n"
              f"Needs ARCHIVAULT_PASSWORD in the environment.")
        return 0
    if not os.environ.get("ARCHIVAULT_PASSWORD"):
        sys.exit("set ARCHIVAULT_PASSWORD first; this script never takes it "
                 "on the command line.")
    if args.reservation_usd is None or args.max_usd is None:
        sys.exit("--confirm requires --reservation-usd and --max-usd; no upstream "
                 "transcription job may bypass a hard spend cap.")
    if args.reservation_usd <= 0 or args.max_usd <= 0:
        sys.exit("--reservation-usd and --max-usd must both be positive.")
    total_reservation = len(pages) * args.reservation_usd
    if total_reservation > args.max_usd + 1e-9:
        sys.exit(f"REFUSING: four-page reservation ${total_reservation:.2f} exceeds "
                 f"the ${args.max_usd:.2f} cap.")

    py = sys.executable
    for p in pages:
        stem = p["file"].replace(".jpg", "")
        pdir = out / stem
        pdir.mkdir(parents=True, exist_ok=True)
        # 1. write the gemini baseline we already have, unmodified
        base = as_volume(p["file"], baseline_for(p["file"]))
        (pdir / "gemini.raw.json").write_text(
            json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
        # 2. Luna, one page, through Archivault
        print(f"\n=== {p['file']} ({p['label']}) ===")
        r = subprocess.run([py, "run_transcription_bakeoff.py", "probe",
                            "--archivault", args.archivault,
                            "--email", args.email, "--model", args.model,
                            "--local-image", str(IMAGES / p["file"]),
                            "--outdir", str(pdir / "luna"),
                            "--reservation-usd", str(args.reservation_usd),
                            "--max-usd", str(args.max_usd), "--ledger", args.ledger,
                            "--confirm"],
                           env=dict(os.environ))
        if r.returncode:
            print(f"  probe failed for {p['file']}; continuing with the rest")
            continue
        print(f"  -> {pdir}")

    print(f"\nAll Luna outputs under {out}. Next, per page:")
    print("  python run_transcription_bakeoff.py segment <raw> --out <segmented>")
    print("  python run_transcription_bakeoff.py score <gemini.seg> <luna.seg> \\")
    print("      --label-a gemini-3.1-pro --label-b luna --outdir <page>/comparison")
    print("  python run_transcription_bakeoff.py report <page>/comparison/"
          "bakeoff_divergent.json --out <page>/side_by_side.html")
    print("\nThen READ THE IMAGES. The screen measures well-formedness, and "
          "fluent fabrication is well formed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
