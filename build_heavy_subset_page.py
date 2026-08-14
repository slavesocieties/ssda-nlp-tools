"""A focused grading page for the rows that actually carry the population.

WHY THIS EXISTS
---------------
Daniel graded the first 20 rows of the 200-row blocked sample. Those rows stand
for 1,995 of 3,013,215 discarded pairs: **0.066%**. The twenty heaviest rows
stand for 1,692,576, or **56.2%** -- 848x more, for the same twenty judgements.

Two causes, both ours. The sample is Horvitz-Thompson weighted, so rows differ in
what they represent by a factor of 134,583, and `build_blocked_labels.py` emitted
them in stratum-NAME order, which scatters weight arbitrarily. And the page shows
**no row numbers at all** -- `data-i` is an invisible attribute -- so "please
grade rows 80 to 104" is not a request anyone can act on without counting 80
items down a scrolling page.

So this builds a page containing only the heavy rows.

TWO PROPERTIES IT MUST HAVE, AND THE REASON FOR EACH
----------------------------------------------------
1.  **Original indices are preserved.** `data-i` carries the row's index in the
    ORIGINAL sample, not its position here. Grades are keyed by position and
    carry no pair identifiers, so renumbering would silently point every answer
    at a different pair. `estimate_blocked_recall.py` can then read this file and
    the first 20 together, against the same locked sample.

2.  **Its own localStorage key and download name.** A shared key lets one
    grading session overwrite another's answers, and Daniel's first 20 are
    already in the `blk` store. This uses `blkH` and
    `blocked_labels_heavy.json`.

It does not touch, rebuild or reorder the delivered page. That page is locked and
his twenty answers are tied to its row positions.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import ssda_nlp_tools.disambiguate as D
from build_targeted_labels import render
from ssda_nlp_tools.volume_geo import load as load_geo


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample",
                    default="production/luna_v3/blocked_labels/blocked_pairs.json")
    ap.add_argument("--graded", default=None,
                    help="grades already returned; those rows are skipped")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--outdir", default="production/luna_v3/blocked_labels_heavy")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--assembled", default="production/luna_v3/assembled",
                    help="corpus the sample was DRAWN from -- the raw assembly")
    a = ap.parse_args(argv)

    sample = json.load(open(a.sample, encoding="utf-8"))
    rows, population = sample["pairs"], sample["population"]

    done = set()
    if a.graded and os.path.exists(a.graded):
        g = json.load(open(a.graded, encoding="utf-8"))
        done = {int(k) for k in (g.get("labels") or g)}
        print(f"skipping {len(done)} rows already graded")

    order = sorted((i for i in range(len(rows)) if i not in done),
                   key=lambda i: -rows[i]["weight"])[:a.top]
    order.sort()                       # present in original order, not by weight
    covered = sum(rows[i]["weight"] for i in order)
    already = sum(rows[i]["weight"] for i in done)

    print(f"sample     : {len(rows)} rows standing for {population:,} pairs")
    print(f"already    : {len(done)} rows = {already:,.0f} ({100*already/population:.3f}%)")
    print(f"this page  : {len(order)} rows = {covered:,.0f} "
          f"({100*covered/population:.1f}%)")
    print(f"together   : {100*(covered+already)/population:.1f}% of the pool")

    # The sample json stores only {entry, id, name} per side. The renderer needs
    # the full mention -- relationships and attributes are what Daniel actually
    # judges on, so a page without them would be worse than the one he has.
    # Rebuild from the corpus the sample was drawn from.
    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    by_key = {(m["_entry"], str(m["_local_id"])): m for m in M}

    picked, missing = [], []
    for i in order:
        r = rows[i]
        ma = by_key.get((r["a"]["entry"], str(r["a"]["id"])))
        mb = by_key.get((r["b"]["entry"], str(r["b"]["id"])))
        if ma is None or mb is None:
            missing.append(i)
            continue
        gap = abs((ma.get("_year") or 0) - (mb.get("_year") or 0))             if ma.get("_year") and mb.get("_year") else 0
        picked.append({"a": ma, "b": mb,
                       "d": {"shared": 0,
                             "na": len({n for _, n in (ma.get("_ctx") or ())}),
                             "nb": len({n for _, n in (mb.get("_ctx") or ())}),
                             "gap": gap}})
    if missing:
        # Silence here would drop rows from the page while the index mapping
        # below still assumed they were present, shifting every later grade.
        raise SystemExit(f"could not resolve {len(missing)} rows in "
                         f"{a.assembled!r}: {missing[:5]}. Wrong corpus?")

    geo = load_geo(a.volumes)
    blurb = (
        f"<b>These are the {len(order)} rows that carry the most weight.</b> "
        f"The 200-row set you already have is a weighted sample: each row stands "
        f"in for a different number of real discarded pairs, from 1 to 134,583. "
        f"The first 20 you kindly graded stand for {already:,.0f} pairs, which is "
        f"{100*already/population:.3f}% of the discarded pool. "
        f"<b>These {len(order)} stand for {covered:,.0f}, about "
        f"{100*covered/population:.0f}% of it.</b> Same question, same scale: "
        f"<b>0</b> certainly different people, <b>100</b> certainly the same. "
        f"Most should still be obvious zeros, and that is a useful answer.")

    html_out = render(picked, geo, tag="blocked_by_prefilter_heavy",
                      title="Discarded pairs: the rows that carry the population",
                      blurb=blurb, store="blkH",
                      filename="blocked_labels_heavy.json")

    # render() numbers rows 0..n-1 by position. Rewrite each data-i to the index
    # in the ORIGINAL sample, so returned grades line up with the locked file.
    for pos, i in enumerate(order):
        html_out = html_out.replace(f"data-i='{pos}'", f"data-i='ORIG{i}'", 1)
    html_out = html_out.replace("data-i='ORIG", "data-i='")

    os.makedirs(a.outdir, exist_ok=True)
    hp = os.path.join(a.outdir, "blocked_pairs_heavy.html")
    open(hp, "w", encoding="utf-8").write(html_out)
    meta = {"source_sample": a.sample,
            "source_sha256": hashlib.sha256(open(a.sample, "rb").read()).hexdigest(),
            "population": population, "rows": order,
            "weight_covered": covered,
            "note": "data-i values are indices into the ORIGINAL 200-row sample"}
    json.dump(meta, open(os.path.join(a.outdir, "heavy_rows.json"), "w",
                         encoding="utf-8"), indent=2)
    print(f"\nwrote {hp}")
    print("The delivered page was not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
