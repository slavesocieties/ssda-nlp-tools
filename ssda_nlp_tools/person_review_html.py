"""One screen per person, instead of one row per pair.

The pairwise page asks 612,495 questions of the form "are these two the same?".
This one asks 13,967 questions of the form "which of these, if any, is the same
as this person?" — the candidates are on screen together with the person's own
details, so the reviewer judges once with full context rather than repeatedly
with none.

Output is byte-compatible with the pairwise page: the same decisions.json shape
({"decisions":[{a,b,decision}]}), so `run_review.py apply` consumes it unchanged.

Rendering is capped (`--limit`). 13,967 screens carrying ~180,000 candidate rows
produces an unusable file — the corpus pairwise page is already 264 MB. Screens
are ordered by best candidate score, so a capped page is the highest-value work
first, which is also how the tail is meant to be managed.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Person review — __TAG__</title><style>
:root{color-scheme:light dark}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
 background:#faf9f7;color:#1a1a1a}
@media(prefers-color-scheme:dark){body{background:#15161a;color:#e8e8ea}}
header{position:sticky;top:0;background:inherit;border-bottom:1px solid #8884;
 padding:12px 20px;z-index:9}
h1{font-size:17px;margin:0 0 2px} .sub{font-size:13px;opacity:.7;margin:0}
.wrap{padding:16px 20px;max-width:1100px}
.person{border:1px solid #8884;border-radius:8px;margin:0 0 18px;overflow:hidden}
.phead{padding:12px 14px;background:#8881}
.pname{font-weight:600;font-size:16px}
.pmeta{font-size:12.5px;opacity:.75;margin-top:3px;font-family:ui-monospace,monospace}
.cand{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;
 padding:10px 14px;border-top:1px solid #8883}
.cand.same{background:rgba(22,163,74,.10)} .cand.different{background:rgba(220,38,38,.08)}
.cand.unsure{background:rgba(234,179,8,.10)}
.cname{font-weight:500} .cmeta{font-size:12px;opacity:.72;font-family:ui-monospace,monospace}
.reasons{font-size:11.5px;opacity:.6;margin-top:2px}
button{font:inherit;padding:4px 9px;margin-left:5px;border:1px solid #8886;
 border-radius:6px;background:transparent;color:inherit;cursor:pointer}
button.on-same{background:#16a34a;color:#fff;border-color:#16a34a}
button.on-diff{background:#dc2626;color:#fff;border-color:#dc2626}
button.on-unsure{background:#ca8a04;color:#fff;border-color:#ca8a04}
kbd{border:1px solid #8886;border-radius:4px;padding:0 4px;font-size:11px}
.score{font-family:ui-monospace,monospace;font-size:12px;opacity:.8;min-width:3.2em;
 display:inline-block;text-align:right}
</style></head><body>
<header>
<h1>Person review — __TAG__</h1>
<p class=sub><b id=done>0</b> of <b id=total>0</b> candidates decided across
 <b id=screens>0</b> people &nbsp;·&nbsp; <button id=dl>Download decisions.json</button>
 &nbsp;<span style="font-size:12px;opacity:.7">keys <kbd>s</kbd> same
 <kbd>d</kbd> different <kbd>u</kbd> unsure <kbd>j</kbd>/<kbd>k</kbd> move</span></p>
</header>
<div class=wrap id=list></div>
<script>
const DATA = __DATA__, TAG = __TAG_JSON__;
const KEY = "ssda-person-review-" + TAG;
let decisions = JSON.parse(localStorage.getItem(KEY) || "{}");
let cur = 0; const rows = [];
DATA.forEach((s, si) => s.candidates.forEach((c, ci) => rows.push({si, ci})));

function detail(d){
  if(!d) return "";
  return Object.entries(d).filter(([k,v]) => v !== null && v !== "" && v !== undefined)
    .map(([k,v]) => k + "=" + v).join("  ");
}
function render(){
  const el = document.getElementById("list");
  el.innerHTML = DATA.map((s, si) => {
    const p = s.person;
    const cands = s.candidates.map((c, ci) => {
      const k = si + ":" + ci, dv = decisions[k] || "";
      return "<div class='cand " + dv + "' id='r" + k + "'>" +
        "<div><span class=cname></span>" +
        "<div class=cmeta></div>" +
        "<div class=reasons></div></div>" +
        "<div><span class=score>" + c.score.toFixed(2) + "</span>" +
        "<button data-k='" + k + "' data-d=same>same</button>" +
        "<button data-k='" + k + "' data-d=different>different</button>" +
        "<button data-k='" + k + "' data-d=unsure>unsure</button></div></div>";
    }).join("");
    return "<div class=person><div class=phead><div class=pname></div>" +
      "<div class=pmeta></div></div>" + cands + "</div>";
  }).join("");
  // set text via textContent so no record value can inject markup
  document.querySelectorAll(".person").forEach((node, si) => {
    const s = DATA[si];
    node.querySelector(".pname").textContent = s.person.name || "(unnamed)";
    node.querySelector(".pmeta").textContent =
      (s.person.entry || "") + " · " + (s.person.id || "") + " · " + detail(s.person.detail);
    node.querySelectorAll(".cand").forEach((cn, ci) => {
      const c = s.candidates[ci];
      cn.querySelector(".cname").textContent = c.name || "(unnamed)";
      cn.querySelector(".cmeta").textContent =
        (c.entry || "") + " · " + (c.id || "") + " · " + detail(c.detail);
      cn.querySelector(".reasons").textContent = (c.reasons || []).join(", ");
    });
  });
  paint();
}
function paint(){
  rows.forEach(({si, ci}) => {
    const k = si + ":" + ci, node = document.getElementById("r" + k);
    if(!node) return;
    node.className = "cand " + (decisions[k] || "");
    node.querySelectorAll("button").forEach(b => {
      b.className = (decisions[k] === b.dataset.d)
        ? {same:"on-same", different:"on-diff", unsure:"on-unsure"}[b.dataset.d] : "";
    });
  });
  document.getElementById("done").textContent =
    Object.values(decisions).filter(v => v === "same" || v === "different").length;
  document.getElementById("total").textContent = rows.length;
  document.getElementById("screens").textContent = DATA.length;
}
function set(k, v){
  decisions[k] = v; localStorage.setItem(KEY, JSON.stringify(decisions)); paint();
}
document.addEventListener("click", e => {
  if(e.target.dataset && e.target.dataset.d) set(e.target.dataset.k, e.target.dataset.d);
});
document.addEventListener("keydown", e => {
  const m = {s:"same", d:"different", u:"unsure"}[e.key];
  if(m && rows[cur]){ const r = rows[cur]; set(r.si + ":" + r.ci, m);
    cur = Math.min(cur + 1, rows.length - 1); focus(); }
  if(e.key === "j"){ cur = Math.min(cur + 1, rows.length - 1); focus(); }
  if(e.key === "k"){ cur = Math.max(cur - 1, 0); focus(); }
});
function focus(){
  const r = rows[cur]; if(!r) return;
  const n = document.getElementById("r" + r.si + ":" + r.ci);
  if(n) n.scrollIntoView({block:"center", behavior:"smooth"});
}
document.getElementById("dl").onclick = () => {
  const out = [];
  DATA.forEach((s, si) => s.candidates.forEach((c, ci) => {
    const v = decisions[si + ":" + ci];
    if(v === "same" || v === "different")
      out.push({a:{entry:s.person.entry, id:s.person.id},
                b:{entry:c.entry, id:c.id}, decision:v, score:c.score});
  }));
  const blob = new Blob([JSON.stringify({tag:TAG, decisions:out}, null, 1)],
                        {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
};
render();
</script></body></html>"""


def render_person_review_html(screens: List[Dict[str, Any]], out_path: str,
                              tag: str = "corpus", limit: int = 500) -> str:
    """Write the per-person review page. `limit` caps how many screens render."""
    shown = screens[:limit] if limit else screens
    data = json.dumps(shown, ensure_ascii=False).replace("<", "\\u003c")
    page = (_PAGE.replace("__DATA__", data)
                 .replace("__TAG_JSON__", json.dumps(tag).replace("<", "\\u003c"))
                 .replace("__TAG__", str(tag).replace("<", "&lt;")))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path
