"""Side-by-side page of the most divergent transcriptions, for adjudication.

Self-contained, no network. Differences are highlighted at word level so the
reviewer's eye goes to the disagreement rather than re-reading two near-identical
paragraphs; on a 900-word page the divergence is often three names.
"""
from __future__ import annotations

import difflib
import html
import json
from typing import Any, Dict, List

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transcription bake-off</title><style>
:root{color-scheme:light dark;font-family:system-ui,Segoe UI,Arial,sans-serif}
body{max-width:1200px;margin:1.4rem auto;padding:0 1rem;line-height:1.5}
h1{font-size:1.3rem;margin:.2rem 0}.sub{color:#888;margin-top:0;font-size:.9rem}
.page{border:1px solid #8884;border-radius:10px;padding:.9rem 1.1rem;margin:1rem 0}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
.sim{font-variant-numeric:tabular-nums;color:#888;font-size:.85rem}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-top:.7rem}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.col h3{font-size:.9rem;margin:.2rem 0;color:#888;font-weight:600}
.txt{font-size:.86rem;white-space:pre-wrap;border-left:2px solid #8883;
     padding-left:.6rem;max-height:26rem;overflow:auto}
ins{background:rgba(22,163,74,.22);text-decoration:none}
del{background:rgba(220,38,38,.20);text-decoration:none}
.note{font-size:.82rem;color:#b45309;margin:.6rem 0 0}
</style></head><body>
<h1>Transcription bake-off &mdash; the pages where the models disagree most</h1>
<p class=sub>Sorted by dissimilarity, most divergent first. Highlighting marks
words present in one transcription and not the other; it says nothing about
which is <em>right</em>. That is the judgement being asked for, and it needs the
manuscript image alongside.</p>
__BODY__
</body></html>"""


def _diff(a: str, b: str):
    """Word-level diff, rendered into both columns."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ta = html.escape(" ".join(aw[i1:i2]))
        tb = html.escape(" ".join(bw[j1:j2]))
        if tag == "equal":
            left.append(ta); right.append(tb)
        else:
            if ta:
                left.append(f"<del>{ta}</del>")
            if tb:
                right.append(f"<ins>{tb}</ins>")
    return " ".join(left), " ".join(right)


def render_bakeoff_html(divergent: List[Dict[str, Any]], out_path: str,
                        label_a: str = "A", label_b: str = "B") -> str:
    blocks = []
    for d in divergent:
        la, lb = _diff(d.get("a", ""), d.get("b", ""))
        blocks.append(
            f"<div class=page><div class=hd><b>{html.escape(str(d['image']))}</b>"
            f"<span class=sim>similarity {d.get('similarity')}</span></div>"
            f"<div class=cols>"
            f"<div class=col><h3>{html.escape(label_a)}</h3><div class=txt>{la}</div></div>"
            f"<div class=col><h3>{html.escape(label_b)}</h3><div class=txt>{lb}</div></div>"
            f"</div></div>")
    if not blocks:
        blocks = ["<p class=note>No divergent pages: the two transcriptions are "
                  "identical on every shared page.</p>"]
    page = _PAGE.replace("__BODY__", "\n".join(blocks))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path
