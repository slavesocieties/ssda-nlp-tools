"""Self-contained review page for labelling pairs on a likelihood scale.

Daniel, 2026-07-29: manual review "on a 0%/25%/50%/75%/100% likelihood of
sameness basis", and "that data can be used to train said model."

Why this is a separate page from review_html.py rather than a change to it:
the binary page produces *constraints* (this pair is the same person, apply it),
this one produces *labels* (this pair is 75% likely, learn from it). Only the
two endpoints mean the same thing in both worlds, and conflating them would let
a 75% judgement silently become a hard must-link. `decisions_to_constraints`
below is deliberately strict about that.

No server, no network, no fonts, no fetch: open the file, press 0-4 or click,
"Download labels.json" saves the result. Progress is mirrored to localStorage so
a closed tab does not lose a session's work.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

# 0/25/50/75/100 as Daniel specified. `value` is what lands in the JSON.
LEVELS = (
    (0,   "0%",   "certainly different", "#dc2626"),
    (25,  "25%",  "probably different",  "#ea580c"),
    (50,  "50%",  "genuinely unclear",   "#d97706"),
    (75,  "75%",  "probably same",       "#65a30d"),
    (100, "100%", "certainly same",      "#16a34a"),
)

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SSDA identity training labels</title><style>
:root{color-scheme:light dark;font-family:system-ui,Segoe UI,Arial,sans-serif}
body{max-width:1000px;margin:1.4rem auto;padding:0 1rem;line-height:1.45}
h1{margin:.2rem 0;font-size:1.35rem}.sub{color:#888;margin-top:0;font-size:.9rem}
.pair{border:1px solid #8884;border-left-width:5px;border-left-color:#8884;
      border-radius:10px;padding:.8rem 1rem;margin:.9rem 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:720px){.cols{grid-template-columns:1fr}}
.col b{font-size:1.05rem}.meta{color:#888;font-size:.8rem}
.detail{font-size:.82rem;margin:.25rem 0 0;padding-left:1rem}
.text{font-size:.78rem;color:#8a8a8a;margin:.4rem 0 0;max-height:6.2rem;
      overflow:auto;border-left:2px solid #8883;padding-left:.5rem;white-space:pre-wrap}
.score{float:right;font-weight:600;font-variant-numeric:tabular-nums}
.reasons{font-size:.78rem;color:#998;margin:.35rem 0 0}
.strat{font-size:.7rem;color:#777;margin:.15rem 0 0;font-family:ui-monospace,monospace}
.btns{margin-top:.6rem;display:flex;gap:.4rem;align-items:center;flex-wrap:wrap}
button{border:1px solid #8886;border-radius:7px;padding:.35rem .75rem;cursor:pointer;
       background:transparent;color:inherit;font-size:.88rem}
button.on{color:#fff}
#bar{position:sticky;top:0;background:Canvas;padding:.6rem 0;
     border-bottom:1px solid #8884;z-index:5}
#dl{background:#3b82f6;color:#fff;border-color:#3b82f6}
kbd{border:1px solid #8886;border-radius:4px;padding:0 .3rem;font-size:.8rem}
.legend{font-size:.78rem;color:#888;margin:.3rem 0 0}
</style></head><body>
<div id=bar><b>SSDA identity training labels</b> &mdash;
<span id=done>0</span>/<span id=total>0</span> labelled
&nbsp;<button id=dl>Download labels.json</button>
<div class=legend>keys: <kbd>0</kbd>&hairsp;<kbd>1</kbd>&hairsp;<kbd>2</kbd>&hairsp;<kbd>3</kbd>&hairsp;<kbd>4</kbd>
 = 0/25/50/75/100&#37; &middot; <kbd>j</kbd>/<kbd>k</kbd> move &middot; <kbd>x</kbd> clear</div></div>
<h1>How likely is it that these are the same person?</h1>
<p class=sub>The number on the right is what the current algorithm scored, shown
for reference only &mdash; it is one of the things being tested, so please judge
the pair on the record, not on the score.</p>
<div id=list></div>
<script>
const PAIRS = __DATA__, LEVELS = __LEVELS__, TAG = __TAG__, FINGERPRINT = __FP__;
// Storage is keyed by PAIR IDENTITY and by a fingerprint of the dataset, not
// by array position under a bare tag. Position-keyed storage silently
// mis-attributes: this sample was regenerated twice in one day, and a reviewer
// who had labelled the earlier build would have reopened the new one to find it
// apparently pre-filled, with every answer attached to a different pair. The
// fingerprint means a different dataset simply starts clean instead of
// colliding.
const KEY = "ssda-labels-" + TAG + "-" + FINGERPRINT;
const pairKey = p => [p.a.entry, p.a.id, p.b.entry, p.b.id].join("|");
// localStorage is not guaranteed on a file:// origin -- Safari refuses it
// outright and Chrome can be configured to. An unguarded read here is a
// top-level statement, so it would throw before a single card rendered and the
// reviewer would get a blank page with no clue why. Persistence is a
// convenience; labelling must work without it.
let STORAGE_OK = true;
function loadLabels() {
  try { return JSON.parse(localStorage.getItem(KEY) || "{}"); }
  catch (e) { STORAGE_OK = false; return {}; }
}
function saveLabels() {
  if (!STORAGE_OK) return;
  try { localStorage.setItem(KEY, JSON.stringify(labels)); }
  catch (e) { STORAGE_OK = false; warnNoStorage(); }
}
function warnNoStorage() {
  if (document.getElementById("nostore")) return;
  const d = document.createElement("div");
  d.id = "nostore"; d.className = "legend";
  d.style.color = "#b45309";
  d.textContent = "This browser will not save progress for a local file, so "
    + "your labels live only in this tab. Download before closing it.";
  document.getElementById("bar").appendChild(d);
}
let labels = loadLabels();
let cursor = 0;
const list = document.getElementById("list");
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function side(x) {
  const det = (x.detail && Object.keys(x.detail).length)
    ? "<ul class=detail>" + Object.entries(x.detail).map(([k,v]) =>
        "<li>" + esc(k) + ": " + esc(Array.isArray(v) ? v.join("; ") : v) + "</li>").join("") + "</ul>"
    : "<div class='detail meta'>no attributes</div>";
  const txt = x.text ? "<div class=text>" + esc(x.text) + "</div>" : "";
  return "<div class=col><b>" + esc(x.name == null ? "(unnamed)" : x.name) + "</b>" +
    "<div class=meta>entry " + esc(x.entry) + " &middot; " + esc(x.id) + "</div>" +
    det + txt + "</div>";
}
function card(p, i) {
  const d = document.createElement("div");
  d.className = "pair"; d.id = "p" + i;
  d.innerHTML = "<span class=score>" + p.score.toFixed(2) + "</span>" +
    "<div class=cols>" + side(p.a) + side(p.b) + "</div>" +
    "<div class=reasons>" + esc((p.reasons || []).join(" &middot; ")) + "</div>" +
    "<div class=strat>" + esc(p.disposition || "") +
      (p.stratum ? " &middot; " + esc(p.stratum) : "") + "</div>" +
    "<div class=btns>" + LEVELS.map(([v, lab, help]) =>
      "<button data-v=" + v + " title='" + esc(help) + "'>" + esc(lab) + "</button>").join("") +
    "</div>";
  d.querySelectorAll("button").forEach(b =>
    b.onclick = () => label(i, Number(b.dataset.v)));
  return d;
}
function label(i, v) {
  const k = pairKey(PAIRS[i]);
  if (v === null) delete labels[k]; else labels[k] = v;
  saveLabels();
  paint(i);
  if (v !== null) { cursor = Math.min(i + 1, PAIRS.length - 1); focus(); }
}
function paint(i) {
  const d = document.getElementById("p" + i);
  const v = labels[pairKey(PAIRS[i])];
  const colour = v === undefined ? "" : (LEVELS.find(l => l[0] === v) || [])[3];
  d.style.borderLeftColor = colour || "";
  d.querySelectorAll("button").forEach(b => {
    const on = Number(b.dataset.v) === v;
    b.className = on ? "on" : "";
    b.style.background = on ? colour : "";
    b.style.borderColor = on ? colour : "";
  });
  document.getElementById("done").textContent = Object.keys(labels).length;
}
function focus() {
  const el = document.getElementById("p" + cursor);
  if (el) { el.scrollIntoView({block: "center", behavior: "smooth"});
            el.style.outline = "2px solid #3b82f6"; setTimeout(()=>el.style.outline="",700); }
}
PAIRS.forEach((p, i) => list.appendChild(card(p, i)));
PAIRS.forEach((_, i) => paint(i));
document.getElementById("total").textContent = PAIRS.length;
// if the READ failed, saveLabels() short-circuits and would never warn, so the
// reviewer would get silent non-persistence -- the worst of both behaviours
if (!STORAGE_OK) warnNoStorage();
document.addEventListener("keydown", e => {
  const k = "01234".indexOf(e.key);
  if (k >= 0) label(cursor, LEVELS[k][0]);
  else if (e.key === "x") label(cursor, null);
  else if (e.key === "j") { cursor = Math.min(cursor + 1, PAIRS.length - 1); focus(); }
  else if (e.key === "k") { cursor = Math.max(cursor - 1, 0); focus(); }
});
document.getElementById("dl").onclick = () => {
  const out = {tag: TAG, scale: "likelihood_same_percent",
    labels: PAIRS.map((p, i) => ({
      a: {entry: p.a.entry, id: p.a.id}, b: {entry: p.b.entry, id: p.b.id},
      names: [p.a.name, p.b.name], score: p.score,
      disposition: p.disposition || null, stratum: p.stratum || null,
      weight: p.weight === undefined ? null : p.weight,
      likelihood: labels[pairKey(p)] === undefined ? null : labels[pairKey(p)]}))};
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "labels.json"; a.click();
};
</script></body></html>"""


def render_likelihood_review_html(pairs: List[Dict[str, Any]], out_path: str,
                                  tag: str = "training", limit: int = 2500) -> str:
    """Write the page. `limit` is a hard cap: the uncapped 701k-pair page came
    out at 230 MB and no browser opens that, so truncation is explicit and the
    highest-scoring work goes first rather than being silently dropped."""
    ordered = sorted(pairs, key=lambda p: -p.get("score", 0))[:limit]
    # Fingerprint of WHICH pairs are present: stable under reordering, different
    # the moment the sample changes. It goes into the localStorage key so a
    # regenerated sample starts clean instead of inheriting labels that were
    # given for different pairs at the same positions.
    ident = sorted("|".join((str(p["a"]["entry"]), str(p["a"]["id"]),
                             str(p["b"]["entry"]), str(p["b"]["id"])))
                   for p in ordered)
    fingerprint = hashlib.sha256("\n".join(ident).encode("utf-8")).hexdigest()[:12]
    # <-escape so no data value can close the script block or inject markup
    enc = lambda o: json.dumps(o, ensure_ascii=False).replace("<", "\\u003c")
    page = (_PAGE.replace("__DATA__", enc(ordered))
                 .replace("__LEVELS__", enc([list(l) for l in LEVELS]))
                 .replace("__TAG__", enc(tag))
                 .replace("__FP__", enc(fingerprint)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path


def labels_to_constraints(labels: Any) -> Dict[str, list]:
    """labels.json -> disambiguation constraints.

    ONLY the endpoints become constraints. 75% is not a must-link: it is a
    training label that says "probably, and I could be wrong". Promoting it
    would re-introduce, under the appearance of human authority, exactly the
    speculative merging the cluster-surname guard exists to prevent.
    """
    if isinstance(labels, str):
        with open(labels, "r", encoding="utf-8") as f:
            labels = json.load(f)
    must, cannot = [], []
    for d in labels.get("labels", []):
        v = d.get("likelihood")
        if v == 100:
            must.append([d["a"], d["b"]])
        elif v == 0:
            cannot.append([d["a"], d["b"]])
    return {"must": must, "cannot": cannot}


def label_summary(labels: Any) -> Dict[str, Any]:
    if isinstance(labels, str):
        with open(labels, "r", encoding="utf-8") as f:
            labels = json.load(f)
    rows = labels.get("labels", [])
    done = [r for r in rows if r.get("likelihood") is not None]
    dist: Dict[int, int] = {}
    for r in done:
        dist[r["likelihood"]] = dist.get(r["likelihood"], 0) + 1
    agree = sum(1 for r in done
                if (r["likelihood"] >= 75) == (r.get("score", 0) >= 0.86))
    return {
        "total": len(rows),
        "labelled": len(done),
        "distribution": dict(sorted(dist.items())),
        "constraints_usable": dist.get(0, 0) + dist.get(100, 0),
        "agrees_with_algorithm": agree,
        "agreement_rate": round(agree / len(done), 4) if done else None,
    }
