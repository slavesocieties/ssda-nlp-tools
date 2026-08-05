#!/usr/bin/env python3
"""run_relationship_review.py — the review queue for the weakest extraction field.

Offline, $0, no network, no key.

    python run_relationship_review.py --limit 200

WHY RELATIONSHIPS
-----------------
Relationships are the weakest dimension the extractor has: entity F1 0.836
against the reference set, versus 0.989 for people and 0.962 for events. They are
also the field the whole social graph is made of, so an error here is not a
missing attribute, it is a wrong edge between two real people.

PROJECT_STATUS said this queue was "built, not yet run". It was not built. The
four review modules in ssda_nlp_tools are all identity/pair review -- "are these
two mentions the same person" -- and the only queue ever produced was for
ethnicity TERMS. Nothing routed a suspect relationship to a human. This does.

WHAT IT SURFACES, and why each is decidable by eye:

  dangling_relationship   the edge points at a person id that is not in the
                          entry. The extractor invented a referent, so the
                          reviewer only has to say what the text actually says.
  null_relationship       `related_person` is null: a relationship type with no
                          object. Repairable without spending anything, but a
                          human should confirm the type was not the thing
                          hallucinated.
  role_contradiction      two people in ONE entry hold mutually exclusive roles
                          (A parent of B while B is parent of A).
  dangling_principal      an event names a principal absent from the entry.

Each row carries the faithful transcription, so the decision needs the page but
not the manuscript. Output is a self-contained HTML page (no server, no network)
plus a JSON index for scripted follow-up.
"""
import argparse
import collections
import glob
import html
import json
import os

from audit_corpus import audit

DIMENSIONS = ("dangling_relationship", "null_relationship",
              "role_contradiction", "dangling_principal")


def collect(entries, buckets, limit_per_kind):
    rows = []
    for kind in DIMENSIONS:
        for issue in (buckets.get(kind) or [])[:limit_per_kind]:
            eid = issue.get("entry")
            e = entries.get(eid) or {}
            people = ((e.get("data") or {}).get("people")) or []
            rows.append({
                "kind": kind,
                "entry": eid,
                "detail": issue.get("detail") or "",
                "text": (e.get("text_faithful") or e.get("normalized") or "")[:1400],
                "people": [{"id": p.get("id"), "name": p.get("name"),
                            "relationships": p.get("relationships") or []}
                           for p in people],
            })
    return rows


def render(rows, counts, total_shown):
    css = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
 background:#faf9f7;color:#1a1a1a}
header{position:sticky;top:0;background:#1a1a1a;color:#fff;padding:14px 22px;z-index:9}
header b{font-size:17px}
.wrap{max-width:1080px;margin:0 auto;padding:22px}
.row{background:#fff;border:1px solid #e2ded7;border-radius:8px;margin:0 0 18px;
 padding:16px 18px}
.kind{display:inline-block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
 padding:3px 8px;border-radius:4px;background:#f0ece4;color:#5a5347;font-weight:600}
.eid{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#6b6459;
 margin-left:8px}
.detail{margin:10px 0;color:#8a3a2a;font-size:13px}
.text{background:#fbf8f2;border-left:3px solid #d8d0c2;padding:10px 12px;margin:10px 0;
 white-space:pre-wrap;font-size:13.5px;max-height:190px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
td,th{border-bottom:1px solid #ece8e1;padding:5px 7px;text-align:left;vertical-align:top}
th{color:#6b6459;font-weight:600;font-size:11.5px;text-transform:uppercase}
.btns{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;padding:6px 13px;border-radius:6px;border:1px solid #cfc8bd;
 background:#fff;cursor:pointer}
button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
@media (prefers-color-scheme:dark){
 body{background:#161513;color:#eceae6}.row{background:#201f1c;border-color:#35322c}
 .text{background:#1a1917;border-left-color:#44403a}.kind{background:#2c2a25;color:#c9c2b4}
 button{background:#242320;color:#eceae6;border-color:#3d3934}
 button.on{background:#eceae6;color:#161513}
 td,th{border-bottom-color:#2e2b26}}
"""
    js = """
const D=JSON.parse(localStorage.getItem('relreview')||'{}');
function mark(i,v){D[i]=v;localStorage.setItem('relreview',JSON.stringify(D));
 document.querySelectorAll('[data-i="'+i+'"] button').forEach(b=>
   b.classList.toggle('on', b.dataset.v===v));
 document.getElementById('n').textContent=Object.keys(D).length;}
function dl(){const b=new Blob([JSON.stringify({tool:'relationship_review',
 decisions:D},null,1)],{type:'application/json'});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download='relationship_decisions.json';a.click();}
window.addEventListener('DOMContentLoaded',()=>{
 for(const [i,v] of Object.entries(D)){
  document.querySelectorAll('[data-i="'+i+'"] button').forEach(b=>
    b.classList.toggle('on', b.dataset.v===v));}
 document.getElementById('n').textContent=Object.keys(D).length;});
"""
    parts = [f"<style>{css}</style>", "<header><b>Relationship review</b> &nbsp;",
             f"{total_shown} rows &nbsp;|&nbsp; decided: <span id=n>0</span> ",
             "&nbsp; <button onclick=dl()>Download decisions.json</button></header>",
             "<div class=wrap>",
             "<p>Relationships are the weakest extracted field (F1 0.836 vs 0.989 "
             "for people). Every row below is a relationship the pipeline could "
             "not resolve against its own entry. The faithful transcription is "
             "included, so the page is enough to decide.</p>",
             "<p><b>drop</b> = the relationship is not in the text. "
             "<b>keep</b> = it is there and we mis-linked it. "
             "<b>unsure</b> = needs the manuscript.</p>"]
    for k in DIMENSIONS:
        if counts.get(k):
            parts.append(f"<p style='font-size:13px;color:#6b6459'>"
                         f"{html.escape(k)}: {counts[k]:,} in corpus</p>")
    for i, r in enumerate(rows):
        ppl = "".join(
            f"<tr><td>{html.escape(str(p['id']))}</td>"
            f"<td>{html.escape(str(p['name'] or ''))}</td>"
            f"<td>{html.escape(json.dumps(p['relationships'], ensure_ascii=False))}</td></tr>"
            for p in r["people"])
        parts.append(
            f"<div class=row data-i='{i}'>"
            f"<span class=kind>{html.escape(r['kind'])}</span>"
            f"<span class=eid>{html.escape(str(r['entry']))}</span>"
            f"<div class=detail>{html.escape(r['detail'])}</div>"
            f"<div class=text>{html.escape(r['text'])}</div>"
            f"<table><tr><th>id</th><th>name</th><th>relationships</th></tr>{ppl}</table>"
            f"<div class=btns>"
            f"<button data-v=drop onclick=\"mark({i},'drop')\">drop</button>"
            f"<button data-v=keep onclick=\"mark({i},'keep')\">keep</button>"
            f"<button data-v=unsure onclick=\"mark({i},'unsure')\">unsure</button>"
            f"</div></div>")
    parts.append("</div><script>" + js + "</script>")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--out", default="production/luna_v3/relationship_review.html")
    ap.add_argument("--limit", type=int, default=200,
                    help="rows per defect kind; the page must stay openable")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json")))
    if not paths:
        raise SystemExit(f"no *.materialized.json under {args.assembled}")
    entries, buckets, _, _, _ = audit(paths)

    counts = {k: len(buckets.get(k) or []) for k in DIMENSIONS}
    rows = collect(entries, buckets, args.limit)
    total = sum(counts.values())

    print(f"{len(entries):,} entries; relationship-dimension defects:")
    for k in DIMENSIONS:
        cap = " (capped)" if counts[k] > args.limit else ""
        print(f"   {k:24s} {counts[k]:6,}{cap}")
    print(f"\n{len(rows):,} rows written of {total:,} total.")
    if len(rows) < total:
        # never let a capped page read as complete
        print(f"   *** {total - len(rows):,} NOT shown: --limit {args.limit} per "
              f"kind. Raise --limit for the rest.")

    open(args.out, "w", encoding="utf-8").write(render(rows, counts, len(rows)))
    idx = os.path.splitext(args.out)[0] + ".json"
    json.dump({"counts": counts, "total": total, "shown": len(rows),
               "limit_per_kind": args.limit,
               "rows": [{k: v for k, v in r.items() if k != "text"} for r in rows]},
              open(idx, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}\n-> {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
