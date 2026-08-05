#!/usr/bin/env python3
"""build_targeted_labels.py — labels for the region nobody has labelled.

Offline, $0, no network, no key.

    python build_targeted_labels.py --size 300

WHY THIS REGION
---------------
The corpus A/B left two models disagreeing and no way to referee them. The
uncalibrated scorer got 24/24 on Daniel's labels and merged a third of the
corpus; the calibrated one is healthy corpus-wide and scores 20/24. His 24 labels
cannot settle it, because NOT ONE of them is the case that separates the models:

    same name, same parish, no shared relationship

There are 29,609 such pairs among lay mentions, 29,146 of them with no shared
associate at all. Both models are guessing across the whole of it.

WHAT THESE PAIRS EXPOSE, AND WHY REAL PAIRS RATHER THAN SYNTHETIC
-----------------------------------------------------------------
The synthetic set cannot probe this: its invented names are absent from the
corpus, so name-rarity -- the strongest term -- is constant across every
synthetic pair. Real pairs carry real frequencies.

And they reveal something the current model throws away. Uncapped name rarity in
this region runs 6.03 to 10.43 nats, a 4-nat spread, but MAX_NAME_LLR caps it at
5.5, so **every one of these 29,609 pairs receives IDENTICAL name evidence**. The
cap exists to honour "no people should be merged strictly based on name
correspondence", and it buys that by flattening a real signal. Whether the
discarded spread actually predicts identity is precisely what these labels would
tell us, so the sample is stratified on the UNCAPPED value.

STRATIFICATION -- one axis at a time, as before:
    rarity quartile   how unusual the shared name is (uncapped)
    associates        0 shared, or 1+ shared
    density           how embedded each side is (thin/one-sided/both-dense)
    date gap          <=5y, 6-20y, 21-40y, >40y

THE PAGE DOES NOT SHOW EITHER MODEL'S ANSWER. Both are recorded in a separate
predictions file for afterwards. Showing them would measure agreement with us
rather than his judgement, and it is his judgement that is the referee.
"""
import argparse
import collections
import glob
import html
import json
import math
import os
import random

import ssda_nlp_tools.disambiguate as D
from ssda_nlp_tools import evidence as E
from ssda_nlp_tools.textmatch import name_similarity, phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo

MAX_BLOCK = 400          # skip the huge clergy blocks; clergy are settled


def gather(M, stats, vol_of):
    blocks = collections.defaultdict(list)
    for i, m in enumerate(M):
        blocks[phonetic_key(m.get("name"))].append(i)
    out = []
    for idxs in blocks.values():
        if len(idxs) > MAX_BLOCK:
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                a, b = M[idxs[x]], M[idxs[y]]
                if a["_entry"] == b["_entry"] or vol_of(a) != vol_of(b):
                    continue
                if name_similarity(a.get("name"), b.get("name")) < 0.95:
                    continue
                if E._clergy(a) or E._clergy(b):
                    continue
                out.append((a, b))
    return out


def describe(a, b, stats):
    _, why = E.network_llr(a, b, stats)
    shared = sum(1 for w in why if w.startswith("shared"))
    na = len({n for _, n in (a.get("_ctx") or ())})
    nb = len({n for _, n in (b.get("_ctx") or ())})
    rarity = -math.log(max(stats.p(a.get("name")), stats.p(b.get("name"))))
    ya, yb = a.get("_year"), b.get("_year")
    gap = abs(ya - yb) if (ya and yb) else None
    return {"shared": shared, "na": na, "nb": nb, "rarity": rarity, "gap": gap}


def stratum(d, cuts):
    r = ("q1" if d["rarity"] < cuts[0] else "q2" if d["rarity"] < cuts[1]
         else "q3" if d["rarity"] < cuts[2] else "q4")
    s = "shared" if d["shared"] else "none"
    lo, hi = min(d["na"], d["nb"]), max(d["na"], d["nb"])
    dens = "both-thin" if hi <= 1 else ("one-sided" if lo <= 1 else "both-dense")
    g = d["gap"]
    gp = ("nodate" if g is None else "<=5y" if g <= 5 else "6-20y" if g <= 20
          else "21-40y" if g <= 40 else ">40y")
    return f"{r}|{s}|{dens}|{gp}"


def render(rows, geo):
    css = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#faf9f7;color:#1a1a1a}
header{position:sticky;top:0;background:#1a1a1a;color:#fff;padding:13px 20px;z-index:9}
.wrap{max-width:1040px;margin:0 auto;padding:20px}
.p{background:#fff;border:1px solid #e2ded7;border-radius:8px;margin:0 0 16px;padding:15px 17px}
.nm{font-size:17px;font-weight:600}
.meta{font-size:12px;color:#6b6459;font-family:ui-monospace,Consolas,monospace;margin:3px 0 9px}
.sides{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.side{background:#fbf8f2;border-left:3px solid #d8d0c2;padding:9px 11px;font-size:13.5px}
.side b{font-size:12px;color:#6b6459;text-transform:uppercase;letter-spacing:.05em}
.rel{margin:2px 0 0 10px;font-size:13px}
.btns{margin-top:12px;display:flex;gap:6px;flex-wrap:wrap}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid #cfc8bd;background:#fff;cursor:pointer}
button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
@media (prefers-color-scheme:dark){body{background:#161513;color:#eceae6}
 .p{background:#201f1c;border-color:#35322c}.side{background:#1a1917;border-left-color:#44403a}
 button{background:#242320;color:#eceae6;border-color:#3d3934}button.on{background:#eceae6;color:#161513}}
"""
    js = """
const D=JSON.parse(localStorage.getItem('tgt')||'{}');
function mk(i,v){D[i]=v;localStorage.setItem('tgt',JSON.stringify(D));
 document.querySelectorAll('[data-i="'+i+'"] button').forEach(b=>b.classList.toggle('on',b.dataset.v==v));
 document.getElementById('n').textContent=Object.keys(D).length;}
function dl(){const b=new Blob([JSON.stringify({tag:'targeted_same_name_same_parish',
 scale:'likelihood_same_percent',labels:D},null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);
 a.download='targeted_labels.json';a.click();}
window.addEventListener('DOMContentLoaded',()=>{for(const[i,v]of Object.entries(D))
 document.querySelectorAll('[data-i="'+i+'"] button').forEach(b=>b.classList.toggle('on',b.dataset.v==v));
 document.getElementById('n').textContent=Object.keys(D).length;});
"""
    P = [f"<style>{css}</style>",
         "<header><b>Same name, same parish</b> &nbsp;",
         f"{len(rows)} pairs &nbsp;|&nbsp; done: <span id=n>0</span> ",
         "&nbsp;<button onclick=dl()>Download labels</button></header><div class=wrap>",
         "<p>Every pair below shares a name and a parish. These are the cases our "
         "two candidate models disagree about, and none of your earlier labels "
         "covers them. Same scale as before: <b>0</b> certainly different, "
         "<b>100</b> certainly the same.</p>",
         "<p style='font-size:13px;color:#6b6459'>Associates, dates and recorded "
         "qualities are shown because you asked for them. Neither model's answer "
         "is shown, deliberately.</p>"]
    for i, r in enumerate(rows):
        def side(m, lbl):
            rels = "".join(
                f"<div class=rel>{html.escape(str(t))}: {html.escape(str(n))}</div>"
                for t, n in sorted(m.get("_ctx") or ())) or \
                "<div class=rel style='color:#a09689'>(no relationships recorded)</div>"
            attrs = ", ".join(f"{k}={m.get(k)}" for k in
                              ("phenotype", "free", "ethnicity", "origin", "age")
                              if m.get(k) is not None) or "(none)"
            return (f"<div class=side><b>{lbl}</b><br>{html.escape(str(m.get('name')))}"
                    f"<div class=meta>{html.escape(str(m['_entry']))} &middot; "
                    f"{m.get('_year') or 'no date'}</div>{rels}"
                    f"<div class=rel style='color:#6b6459'>{html.escape(attrs)}</div></div>")
        a, b, d = r["a"], r["b"], r["d"]
        P.append(
            f"<div class=p data-i='{i}'><div class=nm>{html.escape(str(a.get('name')))}</div>"
            f"<div class=meta>shared associates: {d['shared']} &middot; "
            f"networks {d['na']} vs {d['nb']} &middot; "
            f"gap {d['gap'] if d['gap'] is not None else '?'}y</div>"
            f"<div class=sides>{side(a,'A')}{side(b,'B')}</div><div class=btns>"
            + "".join(f"<button data-v={v} onclick=\"mk({i},{v})\">{v}</button>"
                      for v in (0, 25, 50, 75, 100))
            + "</div></div>")
    P.append("</div><script>" + js + "</script>")
    return "\n".join(P)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--outdir", default="production/luna_v3/targeted")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json"))):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = E.NameStats(M, is_clergy=E._clergy)
    geo = load_geo()
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    pool = gather(M, stats, vol_of)
    print(f"{len(pool):,} same-name same-parish lay pairs")

    ds = [describe(a, b, stats) for a, b in pool]
    rs = sorted(d["rarity"] for d in ds)
    cuts = [rs[int(f * (len(rs) - 1))] for f in (0.25, 0.5, 0.75)]
    print(f"rarity quartile cuts: {[round(c,2) for c in cuts]}")

    cells = collections.defaultdict(list)
    for (a, b), d in zip(pool, ds):
        cells[stratum(d, cuts)].append({"a": a, "b": b, "d": d})
    print(f"{len(cells)} non-empty strata")

    # water-fill, exactly as the training sample does: breadth before depth
    rng = random.Random(args.seed)
    keys = sorted(cells)
    for k in keys:
        rng.shuffle(cells[k])
    picked, taken = [], {k: 0 for k in keys}
    while len(picked) < args.size:
        progressed = False
        for k in keys:
            if len(picked) >= args.size:
                break
            if taken[k] < len(cells[k]):
                picked.append((k, cells[k][taken[k]]))
                taken[k] += 1
                progressed = True
        if not progressed:
            break
    print(f"drew {len(picked)} pairs across {sum(1 for k in keys if taken[k])} strata")

    rows = [dict(r, stratum=k, id=f"tgt-{i:03d}") for i, (k, r) in enumerate(picked)]
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, "targeted_pairs")
    json.dump({"seed": args.seed, "rarity_cuts": cuts,
               "pairs": [{"id": r["id"], "stratum": r["stratum"],
                          "a": {"entry": r["a"]["_entry"], "id": r["a"]["_local_id"],
                                "name": r["a"].get("name")},
                          "b": {"entry": r["b"]["_entry"], "id": r["b"]["_local_id"],
                                "name": r["b"].get("name")},
                          "features": r["d"]} for r in rows]},
              open(base + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # both models' answers, kept OUT of the page
    preds = []
    for r in rows:
        sc = E.score(r["a"], r["b"], stats, geo=geo, vol_of=vol_of)
        preds.append({"id": r["id"], "stratum": r["stratum"],
                      "probability": round(sc["probability"], 4),
                      "decision": sc["decision"],
                      "terms": [[t, round(w, 2)] for t, w in sc["terms"]]})
    json.dump(preds, open(base + ".predictions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(base + ".html", "w", encoding="utf-8").write(render(rows, geo))

    dec = collections.Counter(p["decision"] for p in preds)
    print(f"\ncurrent model on this sample: " +
          ", ".join(f"{k} {v}" for k, v in dec.most_common()))
    print(f"\n-> {base}.html\n-> {base}.json\n-> {base}.predictions.json")
    print("\nThe page does NOT show the model's answer; anchoring would measure "
          "agreement\nwith us rather than his judgement, and his judgement is the "
          "referee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
