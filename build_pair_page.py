"""Build a grading page from ANY list of pairs, keyed so grades map back.

Generalises `build_heavy_subset_page.py`, which is welded to the blocked sample's
format. Takes a json holding `{"population": N, "pairs": [{a:{entry,id}, b:{...}}]}`
and renders a sample of it.

THE ONE THING THIS MUST GET RIGHT
---------------------------------
`render()` bakes each row's POSITION into its buttons as `mk(<i>, <score>)`, and
the downloaded file is that dict. So a subset page numbers its own rows, and the
grades come back as positions rather than as indices into the source list.

That is not a hypothetical. It happened on 2026-08-13: 25 grades came back keyed
0..24, which were also valid rows in the 200-row sample AND exactly the rows
already graded. Merging them raised no error and would have attributed 1.9M pairs
of evidence to the twenty lightest rows.

So this rewrites the onclick index to the SOURCE index and then re-reads its own
rendered HTML to prove it, refusing to write a page whose keys do not match.
Verifying the intent is not verifying the artifact.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re

import ssda_nlp_tools.disambiguate as D
from build_targeted_labels import render
from ssda_nlp_tools.volume_geo import load as load_geo


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True,
                    help="json with {'population': N, 'pairs': [...]}")
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--store", required=True,
                    help="localStorage key -- MUST be unused by any other page")
    ap.add_argument("--download-name", required=True)
    ap.add_argument("--title", default="Pairs for review")
    ap.add_argument("--blurb", default="")
    ap.add_argument("--assembled",
                    default="production/luna_v3/assembled_deduped_sided")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    a = ap.parse_args(argv)

    src = json.load(open(a.pairs, encoding="utf-8"))
    allp = src["pairs"]
    population = src.get("population", len(allp))

    rng = random.Random(a.seed)
    order = sorted(rng.sample(range(len(allp)), min(a.sample, len(allp))))
    each = population / len(order)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    by_key = {(m["_entry"], str(m["_local_id"])): m for m in M}

    picked, missing = [], []
    for i in order:
        r = allp[i]
        ma = by_key.get((r["a"]["entry"], str(r["a"]["id"])))
        mb = by_key.get((r["b"]["entry"], str(r["b"]["id"])))
        if ma is None or mb is None:
            missing.append(i)
            continue
        picked.append({"a": ma, "b": mb,
                       "d": {"shared": 0,
                             "na": len({n for _, n in (ma.get("_ctx") or ())}),
                             "nb": len({n for _, n in (mb.get("_ctx") or ())}),
                             "gap": r.get("gap", 0)}})
    if missing:
        # Dropping a row silently would shift the index mapping for every row
        # after it, which is the same class of bug as the position keying.
        raise SystemExit(f"could not resolve {len(missing)} rows against "
                         f"{a.assembled!r}: {missing[:5]}")

    html_out = render(picked, load_geo(a.volumes), tag=a.store,
                      title=a.title, blurb=a.blurb, store=a.store,
                      filename=a.download_name)

    # Rewrite the onclick index to the SOURCE index, descending so a rewritten
    # value cannot be matched again by a later pass.
    for pos, i in reversed(list(enumerate(order))):
        html_out = html_out.replace(f"mk({pos},", f"mk(@{i},")
        html_out = html_out.replace(f"data-i='{pos}'", f"data-i='@{i}'")
    html_out = html_out.replace("mk(@", "mk(").replace("data-i='@", "data-i='")

    keys = sorted({int(x) for x in re.findall(r"mk\((\d+),", html_out)})
    if keys != order:
        raise SystemExit(f"onclick keys {keys[:5]}... do not match the intended "
                         f"rows {order[:5]}... -- refusing to write a page whose "
                         f"grades would not map back")

    os.makedirs(a.outdir, exist_ok=True)
    hp = os.path.join(a.outdir,
                      a.download_name.replace("_labels_", "_pairs_")
                                     .replace(".json", ".html"))
    open(hp, "w", encoding="utf-8").write(html_out)
    json.dump({"source": a.pairs,
               "source_sha256": hashlib.sha256(
                   open(a.pairs, "rb").read()).hexdigest(),
               "population": population, "rows": order,
               "each_row_stands_for": round(each, 1),
               "note": "grade keys are indices into the SOURCE pairs list"},
              open(os.path.join(a.outdir, "sampled_rows.json"), "w",
                   encoding="utf-8"), indent=2)
    print(f"population {population:,}, sampled {len(order)}, "
          f"each row stands for {each:.1f} pairs")
    print(f"wrote {hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
