#!/usr/bin/env python3
"""build_adjudication_set.py — the pages where human and machine disagree most,
with the manuscript image, side by side, so a person can settle it.

    # key comes from YOUR environment; this script never takes it as an argument
    $env:SSDA_API_URL = "https://hro798f6h5.execute-api.us-east-1.amazonaws.com"
    $env:SSDA_API_KEY = "<your key>"

    python build_adjudication_set.py --top 20            # fetches images
    python build_adjudication_set.py --top 20 --no-fetch # text only, $0, no key

WHY THIS EXISTS
---------------
`run_manual_gold.py` says Gemini and a human disagree on about 6.3% of
characters. It cannot say WHICH of them is right, because settling that needs
the manuscript, and until Daniel stood up the image API on 2026-07-31 we had no
way to reach one. Every disagreement was therefore a fact about two texts rather
than a fact about accuracy.

That matters most for the failure mode we cannot otherwise catch. A transcription
model that invents a plausible entry produces beautifully well-formed prose; the
only thing that exposes it is the page. Luna's confabulation last week was caught
by opening the image and by nothing else.

WHAT IT SELECTS
---------------
The pages ranked by disagreement, drawn from `manual_gold.json`, skipping pages
the transcriber failed outright (those need re-transcription, not adjudication).
High DELETION is ranked first: text a human read and the machine did not is the
shape that hides both dropped entries and invented replacements. It also carries
the 19 pages whose alignment was flagged suspect, since a bad pairing and a bad
transcription look identical in the numbers and different on the page.

ON THE API KEY
--------------
Read from `SSDA_API_KEY` in the environment and never accepted as a command-line
argument, so it cannot land in shell history, a process listing, or a commit.
The presigned URLs the API returns are time-limited but reusable by anyone
holding them, per SSDA's own README, so the HTML embeds downloaded local files
rather than the URLs themselves -- a saved page that stops working in 15 minutes
would be useless anyway, and one that keeps working is a credential to leak.
"""
import argparse
import base64
import html
import json
import os
import sys
import urllib.request

API_URL_ENV = "SSDA_API_URL"
API_KEY_ENV = "SSDA_API_KEY"


def fetch_image(vol, page, outdir, timeout=60):
    """Presigned URL from the API, then the object. Returns a local path."""
    base = os.environ.get(API_URL_ENV, "").rstrip("/")
    key = os.environ.get(API_KEY_ENV)
    if not base or not key:
        raise RuntimeError(
            f"set {API_URL_ENV} and {API_KEY_ENV} in your environment "
            f"(see ssda-image-api/docs/downloading-images.md), or pass --no-fetch")
    obj = f"{vol}-{page}.jpg"
    dest = os.path.join(outdir, obj)
    if os.path.exists(dest):
        return dest
    req = urllib.request.Request(f"{base}/download?key={obj}",
                                 headers={"x-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        meta = json.loads(r.read().decode("utf-8"))
    with urllib.request.urlopen(meta["url"], timeout=timeout) as r:
        data = r.read()
    os.makedirs(outdir, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")


CSS = """
body{font:15px/1.55 system-ui,sans-serif;margin:0;background:#faf9f7;color:#1a1a1a}
header{position:sticky;top:0;background:#1a1a1a;color:#fff;padding:10px 16px;z-index:9}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;margin:16px;padding:14px}
.hd{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.hd b{font-size:17px}.tag{background:#eee;border-radius:4px;padding:2px 7px;font-size:12px}
.warn{background:#fde68a}.bad{background:#fca5a5}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.t{white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;background:#fbfbfb;
   border:1px solid #eee;border-radius:6px;padding:10px;max-height:340px;overflow:auto}
h4{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#666}
img{width:100%;border:1px solid #ccc;border-radius:6px;background:#fff}
.miss{color:#b45309;font-size:13px}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="production/luna_v3/manual_gold.json")
    ap.add_argument("--manual", default="../ssda-openai/json")
    ap.add_argument("--machine", default="../transcriptions/json")
    ap.add_argument("--outdir", default="production/luna_v3/adjudication")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip images entirely: $0, no key, no network")
    args = ap.parse_args(argv)

    from ssda_nlp_tools.manual_gold import human_pages, machine_pages

    gold = json.load(open(args.gold, encoding="utf-8"))
    rows = []
    for vol, rep in gold["volumes"].items():
        failed = set(rep.get("hard_failures") or [])
        suspect = set(rep.get("suspect_alignment") or [])
        for r in rep["pages"]:
            if r["page"] in failed:
                continue            # needs re-transcription, not a human eye
            rows.append({**r, "volume": vol, "suspect": r["page"] in suspect})
    # deletion first: text a human read and the machine did not is the shape
    # that hides both dropped entries and invented replacements
    rows.sort(key=lambda r: (-r["del_rate"], -r["sub_rate"]))
    picked = rows[:args.top]

    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "images")
    texts = {}
    for vol in {r["volume"] for r in picked}:
        h = json.load(open(os.path.join(args.manual, f"{vol}.json"), encoding="utf-8"))
        m = json.load(open(os.path.join(args.machine, f"{vol}.json"), encoding="utf-8"))
        texts[vol] = (human_pages(h), machine_pages(m))

    parts = [f"<style>{CSS}</style>",
             f"<header><b>SSDA adjudication set</b> &mdash; {len(picked)} pages "
             f"where the human and machine transcriptions disagree most. "
             f"Which one matches the manuscript?</header>"]
    fetched = failed_fetch = 0
    for r in picked:
        vol, pg, mpg = r["volume"], r["page"], r.get("machine_page", r["page"])
        hp, mp = texts[vol]
        img_html = ""
        if not args.no_fetch:
            try:
                img_html = f'<img src="{data_uri(fetch_image(vol, mpg, imgdir))}">'
                fetched += 1
            except Exception as e:                       # noqa: BLE001
                failed_fetch += 1
                img_html = (f'<div class="miss">image unavailable: '
                            f'{html.escape(str(e)[:160])}</div>')
        flags = ""
        if r["suspect"]:
            flags += '<span class="tag bad">alignment suspect</span>'
        if r.get("offset"):
            flags += f'<span class="tag warn">realigned {r["offset"]:+d}</span>'
        parts.append(
            f'<div class="card"><div class="hd"><b>{vol} folio {pg}</b>'
            f'<span class="tag">machine page {mpg}</span>'
            f'<span class="tag">deletion {100*r["del_rate"]:.0f}%</span>'
            f'<span class="tag">substitution {100*r["sub_rate"]:.0f}%</span>'
            f'<span class="tag">similarity {r["similarity"]:.2f}</span>{flags}</div>'
            f'{img_html}<div class="cols">'
            f'<div><h4>Human transcription</h4><div class="t">'
            f'{html.escape(hp.get(pg, ""))}</div></div>'
            f'<div><h4>Machine (Gemini) transcription</h4><div class="t">'
            f'{html.escape(mp.get(mpg, ""))}</div></div></div></div>')

    out = os.path.join(args.outdir, "adjudication.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    json.dump(picked, open(os.path.join(args.outdir, "selected.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"selected {len(picked)} of {len(rows)} comparable pages")
    print(f"  suspect alignment among them: {sum(1 for r in picked if r['suspect'])}")
    if args.no_fetch:
        print("  images: skipped (--no-fetch)")
    else:
        print(f"  images fetched {fetched}, failed {failed_fetch}")
    print(f"\n-> {out}")
    if failed_fetch and not os.environ.get(API_KEY_ENV):
        print(f"\n{API_KEY_ENV} is not set. Set it in your shell; this script "
              f"deliberately has no --api-key flag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
