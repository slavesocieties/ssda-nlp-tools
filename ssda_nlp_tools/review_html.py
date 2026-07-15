"""Generate a self-contained HTML page for reviewing borderline merge pairs.

The page needs NO server and NO network: open the file, press s / d / u (same /
different / unsure) or click, then "Download decisions" saves a decisions.json
that apply_review.py feeds back into disambiguation as must/cannot constraints.
Decisions are also kept in localStorage so an interrupted session survives a
browser restart.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, List

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SSDA merge review</title><style>
:root{color-scheme:light dark;font-family:system-ui,Segoe UI,Arial,sans-serif}
body{max-width:900px;margin:1.5rem auto;padding:0 1rem;line-height:1.45}
h1{margin:.2rem 0}.sub{color:#888;margin-top:0}
.pair{border:1px solid #8884;border-radius:10px;padding:.8rem 1rem;margin:.8rem 0}
.pair.same{border-color:#16a34a;background:rgba(22,163,74,.07)}
.pair.different{border-color:#dc2626;background:rgba(220,38,38,.07)}
.pair.unsure{border-color:#d97706;background:rgba(217,119,6,.07)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.col b{font-size:1.05rem}.meta{color:#888;font-size:.8rem}
.detail{font-size:.82rem;margin:.25rem 0 0;padding-left:1rem}
.score{float:right;font-weight:600}.reasons{font-size:.78rem;color:#996;margin:.3rem 0 0}
.btns{margin-top:.6rem;display:flex;gap:.5rem;align-items:center}
button{border:1px solid #8886;border-radius:7px;padding:.35rem .9rem;cursor:pointer;background:transparent;color:inherit}
button.on-same{background:#16a34a;color:#fff;border-color:#16a34a}
button.on-diff{background:#dc2626;color:#fff;border-color:#dc2626}
button.on-unsure{background:#d97706;color:#fff;border-color:#d97706}
#bar{position:sticky;top:0;background:Canvas;padding:.6rem 0;border-bottom:1px solid #8884;z-index:5}
#dl{background:#3b82f6;color:#fff;border-color:#3b82f6}
kbd{border:1px solid #8886;border-radius:4px;padding:0 .3rem;font-size:.8rem}
</style></head><body>
<div id=bar><b>SSDA merge review</b> — <span id=done>0</span>/<span id=total>0</span> decided
&nbsp;<button id=dl>Download decisions.json</button>
<span class=meta>&nbsp; keys: <kbd>s</kbd> same &nbsp;<kbd>d</kbd> different &nbsp;<kbd>u</kbd> unsure &nbsp;<kbd>j</kbd>/<kbd>k</kbd> move</span></div>
<h1>Borderline identity pairs</h1>
<p class=sub>Decide whether each pair of mentions refers to the same historical person.</p>
<div id=list></div>
<script>
const PAIRS = __DATA__;
const KEY = "ssda-review-" + __TAG__;
let decisions = JSON.parse(localStorage.getItem(KEY) || "{}");
let cursor = 0;
const list = document.getElementById("list");
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function card(p, i) {
  const d = document.createElement("div");
  d.className = "pair"; d.id = "p" + i;
  const det = x => (x.detail && Object.keys(x.detail).length)
      ? "<ul class=detail>" + Object.entries(x.detail).map(([k,v]) =>
          "<li>" + esc(k) + ": " + esc(Array.isArray(v) ? v.join("; ") : v) + "</li>").join("") + "</ul>"
      : "<div class='detail meta'>no attributes</div>";
  d.innerHTML = "<span class=score>" + p.score.toFixed(2) + "</span>" +
    "<div class=cols><div class=col><b>" + esc(p.a.name) + "</b>" +
    "<div class=meta>entry " + esc(p.a.entry) + " · " + esc(p.a.id) + "</div>" + det(p.a) + "</div>" +
    "<div class=col><b>" + esc(p.b.name) + "</b>" +
    "<div class=meta>entry " + esc(p.b.entry) + " · " + esc(p.b.id) + "</div>" + det(p.b) + "</div></div>" +
    "<div class=reasons>" + esc((p.reasons || []).join(" · ")) + "</div>" +
    "<div class=btns><button data-d=same>same person</button>" +
    "<button data-d=different>different</button><button data-d=unsure>unsure</button></div>";
  d.querySelectorAll("button").forEach(b =>
    b.onclick = () => decide(i, b.dataset.d));
  return d;
}
function decide(i, val) {
  decisions[i] = val; localStorage.setItem(KEY, JSON.stringify(decisions));
  paint(i); cursor = Math.min(i + 1, PAIRS.length - 1); focus();
}
function paint(i) {
  const d = document.getElementById("p" + i);
  d.className = "pair " + (decisions[i] || "");
  d.querySelectorAll("button").forEach(b => {
    b.className = (decisions[i] === b.dataset.d)
      ? {same: "on-same", different: "on-diff", unsure: "on-unsure"}[b.dataset.d] : "";
  });
  document.getElementById("done").textContent =
    Object.values(decisions).filter(v => v === "same" || v === "different").length;
}
function focus() {
  const el = document.getElementById("p" + cursor);
  if (el) { el.scrollIntoView({block: "center", behavior: "smooth"});
            el.style.outline = "2px solid #3b82f6"; setTimeout(()=>el.style.outline="",700); }
}
PAIRS.forEach((p, i) => list.appendChild(card(p, i)));
PAIRS.forEach((_, i) => paint(i));
document.getElementById("total").textContent = PAIRS.length;
document.addEventListener("keydown", e => {
  if (e.key === "s") decide(cursor, "same");
  else if (e.key === "d") decide(cursor, "different");
  else if (e.key === "u") decide(cursor, "unsure");
  else if (e.key === "j") { cursor = Math.min(cursor + 1, PAIRS.length - 1); focus(); }
  else if (e.key === "k") { cursor = Math.max(cursor - 1, 0); focus(); }
});
document.getElementById("dl").onclick = () => {
  const out = {decisions: PAIRS.map((p, i) => ({
    a: {entry: p.a.entry, id: p.a.id}, b: {entry: p.b.entry, id: p.b.id},
    names: [p.a.name, p.b.name], score: p.score, decision: decisions[i] || "undecided"}))};
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
};
</script></body></html>"""


def render_review_html(review_queue: List[Dict[str, Any]], out_path: str,
                       tag: str = "volume") -> str:
    # <-escape so no data value (e.g. a name containing "</script>" or an
    # HTML fragment) can terminate the script block or inject markup
    data = json.dumps(review_queue, ensure_ascii=False).replace("<", "\\u003c")
    page = _PAGE.replace("__DATA__", data) \
                .replace("__TAG__", json.dumps(tag).replace("<", "\\u003c"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path


def decisions_to_constraints(decisions: Any) -> Dict[str, list]:
    """decisions.json (from the HTML page) -> constraints for disambiguate_volume."""
    if isinstance(decisions, str):
        with open(decisions, "r", encoding="utf-8") as f:
            decisions = json.load(f)
    must, cannot = [], []
    for d in decisions.get("decisions", []):
        pair = [d["a"], d["b"]]
        if d.get("decision") == "same":
            must.append(pair)
        elif d.get("decision") == "different":
            cannot.append(pair)
    return {"must": must, "cannot": cannot}
