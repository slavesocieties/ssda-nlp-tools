"""Segmentation gold-labeling sheets.

Different from run_goldprep (which pre-fills EXTRACTION gold). Here we produce a
self-contained HTML sheet per volume so a historian can *verify the segmentation*
quickly: for each page we show the raw transcription for reference and the
segmenter's proposed entries as cards; the reviewer marks each entry
correct / should-split / merge-into-previous / boundary-wrong, fixes the partial
flag, and can flag "a start is missing here". Those per-entry verdicts are enough
to compute real segmentation precision/recall — the certified accuracy number we
currently lack — without anyone retyping entry text.

Output: <vol>.seggold.html (open in any browser, no server) + <vol>.pred.json
(the predictions, so a downloaded corrections.json can be diffed/scored).
No LLM, no network.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List

from .segment import load_pages, segment_page

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Segmentation gold — __VOL__</title><style>
:root{color-scheme:light dark;font-family:system-ui,Segoe UI,Arial,sans-serif}
body{max-width:1100px;margin:1rem auto;padding:0 1rem;line-height:1.45}
h1{margin:.2rem 0}.sub{color:#888;margin-top:0}
#bar{position:sticky;top:0;background:Canvas;padding:.6rem 0;border-bottom:1px solid #8884;z-index:5}
#dl{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:.5rem 1rem;cursor:pointer}
.page{border:1px solid #8884;border-radius:10px;margin:1.2rem 0;padding:.6rem .9rem}
.page h3{margin:.2rem 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.raw{white-space:pre-wrap;font-size:.8rem;background:#8881;border-radius:8px;padding:.5rem;max-height:520px;overflow:auto}
.entry{border:1px solid #8886;border-radius:8px;padding:.5rem .6rem;margin:.5rem 0}
.entry.correct{border-color:#16a34a;background:rgba(22,163,74,.07)}
.entry.split{border-color:#d97706;background:rgba(217,119,6,.08)}
.entry.merge{border-color:#8b5cf6;background:rgba(139,92,246,.08)}
.entry.bad{border-color:#dc2626;background:rgba(220,38,38,.08)}
.etext{font-size:.82rem;margin:.2rem 0}.eid{font-size:.72rem;color:#888}
.verdict button{border:1px solid #8886;border-radius:6px;padding:.2rem .5rem;margin:.15rem .2rem 0 0;cursor:pointer;background:transparent;color:inherit;font-size:.8rem}
.verdict button.on{background:#3b82f6;color:#fff;border-color:#3b82f6}
.missing{width:100%;margin:.3rem 0;font-size:.8rem}
.note{width:100%;font-size:.8rem;margin-top:.25rem}
.flag{color:#d97706;font-size:.75rem}
kbd{border:1px solid #8886;border-radius:4px;padding:0 .3rem;font-size:.8rem}
</style></head><body>
<div id=bar><b>Segmentation gold — __VOL__</b> · <span id=done>0</span>/<span id=total>0</span> entries judged
&nbsp;<button id=dl>Download corrections.json</button>
<span class=sub>&nbsp; per entry: <kbd>1</kbd> correct <kbd>2</kbd> split <kbd>3</kbd> merge-prev <kbd>4</kbd> wrong</span></div>
<h1>Segmentation review — volume __VOL__</h1>
<p class=sub>For each page, check the proposed entries (right) against the raw transcription (left).
Mark each entry, fix its <b>partial</b> box, and tick "a new entry starts inside this block" if a
split is missing. This produces the ground truth to score the segmenter — no retyping needed.</p>
<div id=list></div>
<script>
const PAGES = __DATA__, VOL = __VOLJSON__;
const KEY = "seggold-" + VOL;
let dec = JSON.parse(localStorage.getItem(KEY) || "{}");
const esc = s => String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const list = document.getElementById("list");
let cursor = null;

function total(){return PAGES.reduce((n,p)=>n+p.entries.length,0);}
function judged(){return Object.values(dec).filter(d=>d&&d.verdict).length;}
function refresh(){document.getElementById("done").textContent=judged();}

PAGES.forEach(pg=>{
  const pd=document.createElement("div"); pd.className="page";
  let ent = pg.entries.map((e,i)=>{
    const key=e.id;
    return `<div class=entry id="e_${esc(key)}">
      <div class=eid>${esc(e.id)} ${e.partial?'<span class=flag>partial</span>':''}</div>
      <div class=etext>${esc(e.text)}</div>
      <div class=verdict data-key="${esc(key)}">
        <button data-v=correct>1 correct</button>
        <button data-v=split>2 split needed</button>
        <button data-v=merge>3 merge into prev</button>
        <button data-v=bad>4 boundary/text wrong</button>
        <label style="margin-left:.5rem"><input type=checkbox data-partial="${esc(key)}"> partial</label>
      </div>
      <input class=note placeholder="note (optional)" data-note="${esc(key)}">
    </div>`;
  }).join("");
  pd.innerHTML=`<h3>${esc(pg.image)} — ${pg.entries.length} proposed entr${pg.entries.length==1?'y':'ies'}
      ${pg.page_type&&pg.page_type!=='register'?`<span class=flag>[${esc(pg.page_type)}]</span>`:''}</h3>
    <div class=cols><div class=raw>${esc(pg.raw)}</div><div>${ent}
      <label class=missing><input type=checkbox data-miss="${esc(pg.image)}">
        ⚠ the proposed entries MISS a record that starts on this page (note where):</label>
      <input class=note data-missnote="${esc(pg.image)}" placeholder="e.g. a new entry starts at 'Domingo, dia...'"></div></div>`;
  list.appendChild(pd);
});

function paint(key){
  const el=document.getElementById("e_"+cssesc(key)); if(!el)return;
  const d=dec[key]||{};
  el.className="entry"+(d.verdict?(" "+d.verdict):"");
  el.querySelectorAll(".verdict button").forEach(b=>b.classList.toggle("on",b.dataset.v===d.verdict));
}
function cssesc(s){return s.replace(/[^a-zA-Z0-9_-]/g,"\\\\$&");}
function setV(key,v){dec[key]=Object.assign({},dec[key],{verdict:v});save();paint(key);cursor=key;}
function save(){localStorage.setItem(KEY,JSON.stringify(dec));refresh();}

list.addEventListener("click",e=>{
  const b=e.target.closest(".verdict button"); if(!b)return;
  setV(b.parentElement.dataset.key,b.dataset.v);
});
list.addEventListener("input",e=>{
  const t=e.target;
  if(t.dataset.note){dec[t.dataset.note]=Object.assign({},dec[t.dataset.note],{note:t.value});save();}
  if(t.dataset.partial){dec[t.dataset.partial]=Object.assign({},dec[t.dataset.partial],{partial_fixed:t.checked});save();}
  if(t.dataset.miss){dec["__miss_"+t.dataset.miss]={missing_start:t.checked};save();}
  if(t.dataset.missnote){dec["__miss_"+t.dataset.missnote]=Object.assign({},dec["__miss_"+t.dataset.missnote]||{},{note:t.value});save();}
});
document.addEventListener("keydown",ev=>{
  if(!cursor||/input|textarea/i.test(ev.target.tagName))return;
  const map={"1":"correct","2":"split","3":"merge","4":"bad"};
  if(map[ev.key])setV(cursor,map[ev.key]);
});
// restore
Object.keys(dec).forEach(k=>{if(!k.startsWith("__"))paint(k);});
PAGES.forEach(pg=>pg.entries.forEach(e=>{
  const d=dec[e.id]||{};
  const pb=document.querySelector(`[data-partial="${cssq(e.id)}"]`); if(pb&&d.partial_fixed)pb.checked=true;
  const nb=document.querySelector(`[data-note="${cssq(e.id)}"]`); if(nb&&d.note)nb.value=d.note;
}));
function cssq(s){return String(s).replace(/"/g,'\\\\"');}
document.getElementById("total").textContent=total(); refresh();
document.getElementById("dl").onclick=()=>{
  const out={volume:VOL, corrections:dec,
    schema:"per-entry verdict: correct|split|merge|bad; partial_fixed; per-page __miss_<img>.missing_start"};
  const blob=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="corrections_"+VOL+".json"; a.click();
};
</script></body></html>"""


def build_sheet(volume_path: str, out_html: str, out_pred: str,
                max_pages: int = 12, vol_id: str = None) -> Dict[str, Any]:
    pages = load_pages(volume_path)
    vol = vol_id or _vol_from_path(volume_path)
    # take a contiguous slice that actually contains records (skip leading blanks)
    picked, i = [], 0
    while i < len(pages) and len(picked) < max_pages:
        img, text = pages[i]
        res = segment_page(text, image=img)
        if res.get("page_type") == "register" or res["entries"] or len(picked) > 0:
            picked.append({"image": img, "raw": text,
                           "page_type": res.get("page_type", "register"),
                           "entries": [{"id": e["id"], "text": e["text"],
                                        "partial": e["partial"]} for e in res["entries"]]})
        i += 1
    n_entries = sum(len(p["entries"]) for p in picked)
    page = (_PAGE.replace("__DATA__", json.dumps(picked, ensure_ascii=False).replace("<", "\\u003c"))
                 .replace("__VOLJSON__", json.dumps(vol))
                 .replace("__VOL__", html.escape(str(vol))))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    with open(out_pred, "w", encoding="utf-8") as f:
        json.dump({"volume": vol, "pages": picked}, f, ensure_ascii=False, indent=1)
    return {"volume": vol, "pages": len(picked), "entries": n_entries}


def _vol_from_path(p: str) -> str:
    import os
    return os.path.basename(p).split(".")[0]


def score_corrections(pred: Any, corrections: Any) -> Dict[str, Any]:
    """Turn a reviewer's corrections_<vol>.json into segmentation metrics.

    Verdict meaning:
      correct — the proposed entry is exactly one real entry (a true positive)
      merge   — this boundary is spurious; the entry should join the previous one
                (an over-segmentation: a false-positive boundary)
      split   — this "entry" actually contains >=2 real entries
                (an under-segmentation: a missed boundary inside)
      bad     — wrong text/boundary (a false positive)
    Per page, `__miss_<image>.missing_start` flags a real entry the segmenter
    never started (a false negative).
    """
    if isinstance(pred, str):
        pred = json.load(open(pred, encoding="utf-8"))
    if isinstance(corrections, str):
        corrections = json.load(open(corrections, encoding="utf-8"))
    corr = corrections.get("corrections", corrections)

    pred_ids = [e["id"] for p in pred["pages"] for e in p["entries"]]
    n_pred = len(pred_ids)
    from collections import Counter
    tally = Counter()
    judged = 0
    for eid in pred_ids:
        d = corr.get(eid) or {}
        v = d.get("verdict")
        if v:
            judged += 1
            tally[v] += 1
    missing_pages = sum(1 for k, v in corr.items()
                        if k.startswith("__miss_") and isinstance(v, dict)
                        and v.get("missing_start"))

    correct = tally.get("correct", 0)
    over = tally.get("merge", 0)         # spurious boundaries (false positives)
    under = tally.get("split", 0)        # blocks hiding >=2 real entries
    wrong = tally.get("bad", 0)
    fp = over + wrong
    # precision over the entries actually judged
    precision = correct / judged if judged else None
    # approximate recall lower bound: real entries found vs found+missed. Each
    # 'split' block hides >=1 extra real entry; each flagged page >=1 missed start.
    found = correct + under                  # blocks that cover >=1 real start
    missed = under + missing_pages           # >=1 extra inside splits + page misses
    recall_approx = found / (found + missed) if (found + missed) else None
    return {
        "predicted_entries": n_pred, "judged": judged,
        "exact_correct": correct,
        "precision": round(precision, 3) if precision is not None else None,
        "recall_approx": round(recall_approx, 3) if recall_approx is not None else None,
        "over_splits_merge": over, "under_splits": under, "wrong": wrong,
        "false_positive_boundaries": fp,
        "pages_with_missing_starts": missing_pages,
        "note": "precision is exact; recall is an approximate lower bound "
                "(split/missing flags imply >=1 missed entry, exact count unknown)",
    }


def format_score(vol: str, s: Dict[str, Any]) -> str:
    p = f"{s['precision']:.1%}" if s["precision"] is not None else "n/a"
    r = f"~{s['recall_approx']:.1%}" if s["recall_approx"] is not None else "n/a"
    return (f"{vol}: {s['judged']}/{s['predicted_entries']} judged | "
            f"precision {p} ({s['exact_correct']} exact) | recall {r} | "
            f"over-split {s['over_splits_merge']}, under-split {s['under_splits']}, "
            f"wrong {s['wrong']}, pages-missing-a-start {s['pages_with_missing_starts']}")
