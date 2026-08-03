#!/usr/bin/env python3
"""sample_images.py — a stratified image sample across the whole collection.

    python sample_images.py                        # PLAN only: $0, no key, no network
    python sample_images.py --fetch --images 200   # needs SSDA_API_KEY in env

Daniel, 2026-08-03, on the image API: "you could also use this in conjunction
with the volume metadata (volumes.json) to target all images from specific
volumes and/or create a larger random sample. Feel free to use in this way too,
but don't literally scrape the whole bucket (or even download 10ks of images)."

So this plans a SMALL, DELIBERATELY SPREAD sample and enforces its own ceiling.

WHY A SPREAD SAMPLE AND NOT MORE OF WHAT WE HAVE
------------------------------------------------
volumes.json describes 3,900 volumes holding 750,527 images. The collection is
2,829 Brazilian volumes against 397 Cuban, and 2,827 Portuguese against 1,060
Spanish. Our five delivered volumes are Spanish and Cuban almost throughout.

Every quality number this project has -- 6.3% transcription substitution, 98.2%
segmentation against human entries -- was measured on that Spanish Cuban slice,
and the three segmenter bugs found this week were all found there too. The
"mill" fix is 18th-century Spanish orthography; the markdown-table fix came from
a Colombian volume. None of them was discovered on Brazilian material, which is
most of the archive.

That is a generalisation gap, not a coverage complaint. A stratified sample is
how you find out whether the pipeline works on the collection rather than on the
corner of it we happen to have processed.

THE CEILING IS ENFORCED HERE, not left to the caller
----------------------------------------------------
`MAX_IMAGES` is a hard stop and `--images` above it fails rather than warns.
Downloading is opt-in (`--fetch`), the plan is free and needs no key, and the
key is read from the environment only -- there is deliberately no --api-key
flag, so it cannot reach shell history or a commit.
"""
import argparse
import base64
import json
import os
import random
import sys
import urllib.request
from collections import Counter, defaultdict

API_URL_ENV, API_KEY_ENV = "SSDA_API_URL", "SSDA_API_KEY"

# Daniel's limit, made mechanical. Raising it is a conversation, not a flag.
MAX_IMAGES = 1000
DEFAULT_IMAGES = 200


def era(fields):
    d = str(fields.get("start_date") or "")
    if len(d) >= 4 and d[:4].isdigit():
        y = int(d[:4])
        return f"{y // 50 * 50}s"
    return "undated"


def stratum(fields):
    lang = fields.get("language")
    lang = lang[0] if isinstance(lang, list) and lang else (lang or "?")
    country = fields.get("country")
    country = country[0] if isinstance(country, list) and country else (country or "?")
    typ = fields.get("type")
    typ = typ[0] if isinstance(typ, list) and typ else (typ or "?")
    return f"{country}|{lang}|{typ}|{era(fields)}"


def n_images(fields):
    v = fields.get("images")
    if isinstance(v, (int, float)):
        return int(v)
    return int(v) if isinstance(v, str) and v.isdigit() else 0


def plan(volumes, n_images_total, per_volume, seed, exclude):
    """Water-fill across strata, exactly as the pair sampler does: every stratum
    gets the same depth until it runs out or the budget does. Maximises the
    number of distinct kinds of book seen, which is the whole point."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for v in volumes:
        vid = str(v.get("id") or "")
        f = v.get("fields") or {}
        if not vid or vid in exclude or n_images(f) <= 0:
            continue
        by[stratum(f)].append((vid, n_images(f)))

    keys = sorted(by)
    for k in keys:
        rng.shuffle(by[k])
    picked, cursor, budget = [], {k: 0 for k in keys}, n_images_total
    while budget > 0:
        progressed = False
        for k in keys:
            if budget <= 0:
                break
            i = cursor[k]
            if i >= len(by[k]):
                continue
            vid, total = by[k][i]
            cursor[k] += 1
            progressed = True
            take = min(per_volume, total, budget)
            pages = sorted(rng.sample(range(1, total + 1), take))
            picked.append({"volume": vid, "stratum": k, "volume_images": total,
                           "keys": [f"{vid}-{p:04d}.jpg" for p in pages]})
            budget -= take
        if not progressed:
            break
    return picked


def fetch(key, outdir, timeout=60):
    base = os.environ.get(API_URL_ENV, "").rstrip("/")
    api = os.environ.get(API_KEY_ENV)
    if not base or not api:
        raise RuntimeError(f"set {API_URL_ENV} and {API_KEY_ENV} in your "
                           f"environment; this script has no --api-key flag")
    dest = os.path.join(outdir, key)
    if os.path.exists(dest):
        return dest, True
    req = urllib.request.Request(f"{base}/download?key={key}",
                                 headers={"x-api-key": api})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        meta = json.loads(r.read().decode("utf-8"))
    with urllib.request.urlopen(meta["url"], timeout=timeout) as r:
        data = r.read()
    os.makedirs(outdir, exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(data)
    return dest, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--images", type=int, default=DEFAULT_IMAGES,
                    help=f"total images to sample (hard ceiling {MAX_IMAGES})")
    ap.add_argument("--per-volume", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--outdir", default="production/luna_v3/collection_sample")
    ap.add_argument("--exclude-delivered", action="store_true", default=True,
                    help="skip the five volumes we have already processed")
    ap.add_argument("--fetch", action="store_true",
                    help="actually download; needs SSDA_API_KEY in the environment")
    args = ap.parse_args(argv)

    if args.images > MAX_IMAGES:
        ap.error(f"--images {args.images} exceeds the {MAX_IMAGES} ceiling. "
                 f"Daniel asked us not to bulk-download; raising this is a "
                 f"conversation, not a flag.")

    volumes = json.load(open(args.volumes, encoding="utf-8"))
    exclude = {"176899", "201991", "29597", "375062", "701054"} if args.exclude_delivered else set()
    total_images = sum(n_images(v.get("fields") or {}) for v in volumes)
    print(f"collection: {len(volumes):,} volumes, {total_images:,} images")
    print(f"sampling {args.images} images ({100*args.images/total_images:.3f}% "
          f"of the bucket), <= {args.per_volume} per volume\n")

    picked = plan(volumes, args.images, args.per_volume, args.seed, exclude)
    keys = [k for p in picked for k in p["keys"]]
    print(f"planned {len(keys)} images across {len(picked)} volumes, "
          f"{len({p['stratum'] for p in picked})} strata")
    by_country = Counter(p["stratum"].split("|")[0] for p in picked)
    by_lang = Counter(p["stratum"].split("|")[1] for p in picked)
    print(f"   country  {dict(by_country.most_common(6))}")
    print(f"   language {dict(by_lang.most_common(5))}")

    os.makedirs(args.outdir, exist_ok=True)
    manifest = os.path.join(args.outdir, "sample_plan.json")
    json.dump({"seed": args.seed, "images": len(keys), "volumes": picked},
              open(manifest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {manifest}")

    if not args.fetch:
        print("   PLAN ONLY. Nothing downloaded, no key read. Re-run with "
              "--fetch once SSDA_API_KEY is set.")
        return 0

    got = cached = failed = 0
    for k in keys:
        try:
            _, was_cached = fetch(k, os.path.join(args.outdir, "images"))
            got += 1
            cached += was_cached
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"   ! {k}: {str(e)[:110]}")
    print(f"\ndownloaded {got - cached}, already present {cached}, failed {failed}")
    print(f"-> {os.path.join(args.outdir, 'images')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
