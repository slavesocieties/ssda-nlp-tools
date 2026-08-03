#!/usr/bin/env python3
"""build_synthetic_set.py — corner-case pairs for Daniel to label.

Offline, $0, no network, no key.

    python build_synthetic_set.py --size 300

Daniel, 2026-08-03: "rather than depend on this limited set of live data,
perhaps a better option would be to create an artificial set of people records
with more nuanced (but realistic) sets of characteristics. That way, I can
inform intended behavior on true corner cases rather than fairly obvious
0/100s. Have Claude generate a few hundred of those and I'll work through them."

Writes the pairs, a labelling page in the same 0/25/50/75/100 format as the live
sample, and -- separately -- what the CURRENT algorithm decides for each one.

WHY OUR PREDICTION IS COMPUTED BUT NOT SHOWN
--------------------------------------------
It is written to `*.predictions.json`, never into the HTML. Showing it would
anchor the judgement, and the whole value of this set is an opinion formed
independently of ours. Afterwards the two can be compared and the disagreements
are the interesting rows.

The pairs also carry `expected: null` rather than a guess, for the same reason.
"""
import argparse
import json
import os
import sys
from collections import Counter

from ssda_nlp_tools.disambiguate import (MIN_SIGNALS_FOR_ANY_MERGE,
                                         corroborating_signals, lifespan_conflict,
                                         pair_score, surname_tier_allows)
from ssda_nlp_tools.synthetic_pairs import FAMILIES, generate

AUTO = 0.86


def to_mention(p, entry_id):
    """Synthetic person -> the shape the scorer expects."""
    m = {k: v for k, v in p.items()
         if k not in ("relations", "year") and v is not None}
    m["_entry"] = entry_id
    m["_local_id"] = "P01"
    m["_year"] = p.get("year")
    m["_register"] = "SYN"
    m["_ctx"] = {(r, n) for r, n in (p.get("relations") or [])}
    m["_descendants"] = set()
    m["_unique_sacrament"] = False
    return m


def predict(pair):
    a = to_mention(pair["a"], f"SYN-{pair['id']}-A")
    b = to_mention(pair["b"], f"SYN-{pair['id']}-B")
    allowed, tier = surname_tier_allows(a, b)
    score, reasons = pair_score(a, b, a["_ctx"], b["_ctx"])
    signals = corroborating_signals(a, b)
    merge = allowed and score >= AUTO and len(signals) >= MIN_SIGNALS_FOR_ANY_MERGE
    return {"id": pair["id"], "family": pair["family"],
            "would_merge": bool(merge), "score": round(score, 3),
            "blocked_by": None if allowed else tier,
            "signals": signals, "reasons": reasons,
            "lifespan_conflict": lifespan_conflict(a, b)}


def _fmt(p):
    bits = []
    for k in ("phenotype", "free", "ethnicity", "origin", "age", "occupation"):
        v = p.get(k)
        if v is not None:
            bits.append(f"{k}: {v}")
    for r, n in p.get("relations") or []:
        bits.append(f"{r}: {n}")
    if p.get("year"):
        bits.append(f"year: {p['year']}")
    return bits


CSS = """
body{font:15px/1.55 system-ui,sans-serif;margin:0;background:#faf9f7;color:#1a1a1a}
header{position:sticky;top:0;background:#1a1a1a;color:#fff;padding:10px 16px;z-index:9}
header b{font-size:16px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;margin:14px;padding:14px}
.fam{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:4px;
     padding:2px 8px;font-size:12px;margin-bottom:6px}
.q{font-size:15px;margin:6px 0 10px;color:#111}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.p{background:#fbfbfb;border:1px solid #eee;border-radius:6px;padding:10px}
.p h4{margin:0 0 6px;font-size:15px}
.p ul{margin:0;padding-left:16px;font-size:13px;color:#444}
.note{color:#666;font-size:12px;margin-top:8px;font-style:italic}
.btns{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}
button.l{border:1px solid #bbb;background:#fff;border-radius:6px;padding:6px 14px;
         cursor:pointer;font-size:14px}
button.l.sel{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
#bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
#dl{background:#fff;color:#1a1a1a;border:0;border-radius:6px;padding:6px 12px;cursor:pointer}
"""

JS = """
const PAIRS = __PAIRS__;
const KEY = "ssda-synth-__FP__";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { labels = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch(e){} }
function esc(s){ return (s==null?"":String(s)).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function done(){
  document.getElementById("done").textContent = Object.keys(labels).length;
  document.getElementById("total").textContent = PAIRS.length;
}
function pick(i, v){
  labels[PAIRS[i].id] = v; save(); render(i); done();
}
function render(i){
  const p = PAIRS[i], el = document.getElementById("p"+i);
  const cur = labels[p.id];
  el.querySelectorAll("button.l").forEach(b=>{
    b.className = "l" + (String(b.dataset.v)===String(cur) ? " sel" : "");
  });
}
window.onload = function(){
  const list = document.getElementById("list");
  PAIRS.forEach((p,i)=>{
    const d = document.createElement("div");
    d.className = "card"; d.id = "p"+i;
    const side = (x,t)=>'<div class="p"><h4>'+esc(t)+': '+esc(x.name)+'</h4><ul>'+
      (x._lines||[]).map(l=>'<li>'+esc(l)+'</li>').join('')+'</ul></div>';
    d.innerHTML = '<span class="fam">'+esc(p.family)+'</span>'+
      '<div class="q">'+esc(p.question)+'</div>'+
      '<div class="cols">'+side(p.a,"A")+side(p.b,"B")+'</div>'+
      (p.note ? '<div class="note">'+esc(p.note)+'</div>' : '')+
      '<div class="btns">'+[0,25,50,75,100].map(v=>
        '<button class="l" data-v="'+v+'" onclick="pick('+i+','+v+')">'+v+'%</button>').join('')+
      '</div>';
    list.appendChild(d); render(i);
  });
  done();
  document.getElementById("dl").onclick = function(){
    const out = {tag:"synthetic", scale:"likelihood_same_percent",
      labels: PAIRS.map(p=>({id:p.id, family:p.family, question:p.question,
        a:p.a, b:p.b,
        likelihood: labels[p.id]===undefined ? null : labels[p.id]}))};
    const blob = new Blob([JSON.stringify(out,null,1)], {type:"application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "synthetic_labels.json"; a.click();
  };
};
"""


def render_html(pairs, path, fingerprint):
    for p in pairs:
        p["a"]["_lines"] = _fmt(p["a"])
        p["b"]["_lines"] = _fmt(p["b"])
    html = (f"<style>{CSS}</style>"
            f"<header><div id=bar><b>SSDA synthetic corner cases</b>"
            f"<span><span id=done>0</span>/<span id=total>0</span> labelled</span>"
            f"<button id=dl>Download synthetic_labels.json</button>"
            f"<span style='font-size:12px;opacity:.75'>"
            f"0 = certainly different &middot; 100 = certainly the same"
            f"</span></div></header><div id=list></div>"
            f"<script>{JS.replace('__PAIRS__', json.dumps(pairs, ensure_ascii=False)).replace('__FP__', fingerprint)}</script>")
    open(path, "w", encoding="utf-8").write(html)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--outdir", default="production/luna_v3/synthetic")
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args(argv)

    pairs = generate(args.size, args.seed, args.families)
    os.makedirs(args.outdir, exist_ok=True)
    fam = Counter(p["family"] for p in pairs)
    print(f"{len(pairs)} pairs across {len(fam)} families")
    for k, c in sorted(fam.items()):
        print(f"   {c:4d}  {k}")

    preds = [predict(p) for p in pairs]
    merges = sum(1 for x in preds if x["would_merge"])
    print(f"\nwhat the CURRENT algorithm would do: {merges} merge, "
          f"{len(preds) - merges} do not")
    by_fam = {}
    for x in preds:
        d = by_fam.setdefault(x["family"], [0, 0])
        d[0 if x["would_merge"] else 1] += 1
    for k in sorted(by_fam):
        m, n = by_fam[k]
        print(f"   {k:20s} merge {m:3d}  refuse {n:3d}")

    base = os.path.join(args.outdir, "synthetic_pairs")
    json.dump({"seed": args.seed, "pairs": pairs},
              open(base + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(preds, open(base + ".predictions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    html = render_html(pairs, base + ".html", str(args.seed))
    print(f"\n-> {base}.json\n-> {base}.predictions.json\n-> {html}")
    print("\nThe page does NOT show our prediction. Anchoring the judgement "
          "would waste the exercise;\nthe predictions file is for comparing "
          "afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
