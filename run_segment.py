#!/usr/bin/env python3
"""run_segment.py — parse Archivault transcriptions into sacramental entries (Task 3).

    python run_segment.py INPUT [MORE ...] [--out segmented.json] [--per-image]
                          [--eval REFERENCE.json] [--structural]

INPUT may be an Archivault volume JSON ([{images:[{file,transcription}]}]), a
single {file,transcription} JSON, or the paired-example .md format. Output
matches the paired examples exactly:

    {"image": "...jpg", "entries": [{"id": "<stem>-NN", "text": "...", "partial": false}]}

Default is VOLUME mode: cross-page partial entries are stitched and each entry
lists its source_images. --per-image keeps the page-independent gold format.
Deterministic — $0.00 per image, no API calls. Pages with low segmentation
confidence are listed so ONLY those need an LLM fallback.

    --eval REFERENCE.json   score against reference entries (fuzzy, space-insensitive)
    --structural            validate entry counts against the register's margin numbers
"""
import argparse
import json

from ssda_nlp_tools.segment import load_pages, segment_page, segment_volume
from ssda_nlp_tools.segeval import (
    evaluate_segmentation, format_segeval, load_reference_entries, margin_number_check)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", metavar="PATH", help="write segmentation JSON")
    ap.add_argument("--per-image", action="store_true",
                    help="page-independent output (gold-pair format), no stitching")
    ap.add_argument("--eval", metavar="REF", help="score against a reference volume")
    ap.add_argument("--structural", action="store_true",
                    help="check entry counts against margin numbers")
    args = ap.parse_args(argv)

    pages = []
    for src in args.inputs:
        pages.extend(load_pages(src))
    print(f"{len(pages)} page(s) loaded from {len(args.inputs)} input(s)")

    if args.per_image:
        result = [segment_page(t, image=img) for img, t in pages]
        n = sum(len(r["entries"]) for r in result)
        low = [r["image"] for r in result if r["confidence"] < 0.7]
        print(f"entries: {n} across {len(result)} page(s); "
              f"low-confidence pages: {len(low)} {low or ''}")
        payload = result
    else:
        res = segment_volume(pages)
        s = res["stats"]
        print(f"entries: {s['entries']}  cross-page stitched: {s['cross_page']}  "
              f"still partial: {s['still_partial']}")
        print(f"low-confidence pages (route these to the LLM fallback): "
              f"{s['low_confidence_pages']} {res['low_confidence'] or ''}")
        payload = res

    if args.structural:
        per_image = payload if args.per_image else payload["per_image"]
        mc = margin_number_check(pages, per_image)
        print(f"structural check vs margin numbers: {mc['agree']}/{mc['pages']} pages "
              f"({mc['agreement']:.1%})")
        for r in mc["rows"]:
            if not r["ok"]:
                print(f"   MISMATCH {r['image']}: margins={r['margin_numbers']} "
                      f"expected {r['expected_starts']} got {r['predicted_starts']}")

    if args.eval:
        ref = load_reference_entries(args.eval)
        entries = ([e for r in payload for e in r["entries"]] if args.per_image
                   else payload["entries"])
        rep = evaluate_segmentation(ref, entries)
        print(format_segeval(rep, args.eval))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
