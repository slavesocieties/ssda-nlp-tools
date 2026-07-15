#!/usr/bin/env python3
"""run_seggold.py — generate segmentation gold-labeling sheets for volumes.

    python run_seggold.py VOL1.json VOL2.json ... [--outdir gold_sheets]
                          [--pages 12]

Writes, per volume, a self-contained HTML review sheet (open in any browser, no
server) plus the raw predictions JSON. A historian marks each proposed entry
correct / split / merge / wrong and flags missing starts; the downloaded
corrections_<vol>.json is enough to score real segmentation precision/recall.
No LLM, no network. Pick a DIVERSE set of volumes (different regions, eras,
numbering styles) so the resulting gold certifies accuracy across formats, not
just one.
"""
import argparse
import json
import os

from ssda_nlp_tools.seggold import build_sheet, score_corrections, format_score


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volumes", nargs="*", help="transcription JSON files")
    ap.add_argument("--outdir", default="gold_sheets")
    ap.add_argument("--pages", type=int, default=12, help="pages to sample per volume")
    ap.add_argument("--ids", nargs="*", default=None, help="override volume ids (parallel to volumes)")
    ap.add_argument("--score", nargs=2, metavar=("PRED.json", "corrections.json"),
                    help="score a completed corrections file against its predictions")
    args = ap.parse_args(argv)

    if args.score:
        s = score_corrections(args.score[0], args.score[1])
        vol = json.load(open(args.score[0], encoding="utf-8")).get("volume", "?")
        print(format_score(vol, s))
        print(json.dumps(s, indent=1))
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    total_entries = 0
    for i, v in enumerate(args.volumes):
        vid = args.ids[i] if args.ids and i < len(args.ids) else None
        stem = vid or os.path.basename(v).split(".")[0]
        info = build_sheet(v, os.path.join(args.outdir, f"{stem}.seggold.html"),
                           os.path.join(args.outdir, f"{stem}.pred.json"),
                           max_pages=args.pages, vol_id=vid)
        total_entries += info["entries"]
        print(f"  {info['volume']}: {info['pages']} pages, {info['entries']} entries "
              f"-> {stem}.seggold.html")
    print(f"\n{len(args.volumes)} sheets, ~{total_entries} entries to review "
          f"-> {args.outdir}/")
    print("Open each .seggold.html in a browser, mark the entries, click "
          "'Download corrections.json'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
