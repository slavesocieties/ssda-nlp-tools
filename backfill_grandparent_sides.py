"""Recover the family side of already-extracted grandparents, for free.

WHY THIS EXISTS
---------------
Daniel Genkins, 2026-08-10, on the merge scorer being unable to tell one record
naming the maternal pair from a genuine four-grandparent contradiction:

    "This is a question that can/should be resolved upstream. Maternal/paternal
     grandparents should be labeled differently when extracted."

The extraction schema now does that (`maternal grandparent` / `paternal
grandparent`; see vocab_extensions.json and ssda_nlp_tools/grandparent_side.py),
which fixes every record extracted from here on. It does nothing for the 2,268
grandparent relations already delivered across the seven volumes -- and
re-extracting 6,794 entries to obtain a field the transcription already contains
would be paying twice for the same information.

The side IS in the transcription. 1,400 of the 1,431 delivered entries carrying
a grandparent edge (97.8%) state it outright, in a handful of fixed formulae.
This script reads it back off the entries' own text. No API calls, no cost.

WHAT IT WILL AND WILL NOT DO
----------------------------
It relabels ONLY the edge pointing at the grandparent, and only when the name
falls unambiguously inside exactly one side clause. Everything else keeps the
plain `grandparent` label, which stays legal precisely so there is somewhere
honest to put it. Measured over the delivered corpus: 91.7% resolved, with the
residue split between records that genuinely never say which side and a handful
of extraction defects (a grandparent edge pointing at the godparent, which must
NOT inherit whichever side clause happens to sit nearest).

It writes nothing without --out. There is no in-place mode: these are delivered
production records, and the diff is worth reading before it replaces them.

    python backfill_grandparent_sides.py production/luna_final_7vol/assembled/*.json
    python backfill_grandparent_sides.py <same> --out production/sided_7vol
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

from ssda_nlp_tools.grandparent_side import (MATERNAL, PATERNAL, REASONS,
                                             label_entry, side_clauses)


def process(path, out_dir=None, samples=0):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    entries = doc.get("entries") if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: no entries array")

    stats = collections.Counter()
    shown = []
    labelled = []
    audit = []               # every grandparent that kept the unsided label
    for entry in entries:
        if not isinstance(entry, dict):
            labelled.append(entry)
            continue
        new_entry, s = label_entry(entry)
        detail = s.pop("unresolved_detail", [])
        stats.update(s)
        if s["seen"]:
            stats["entries_with_grandparents"] += 1
        if s["unresolved"]:
            stats["entries_with_unresolved"] += 1
            sides = [c[0].split()[0] for c in side_clauses(
                entry.get("normalized") or entry.get("text_faithful") or "")]
            for name, reason in detail:
                audit.append({"entry": entry.get("id"), "grandparent": name,
                              "reason": reason, "side_clauses": sides})
            if len(shown) < samples:
                shown.append((entry.get("id"), sides))
        labelled.append(new_entry)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, os.path.basename(path))
        if isinstance(doc, dict):
            doc["entries"] = labelled
            # A record of what touched the file, in the same spirit as the
            # provenance block the assembler already writes. The reason
            # breakdown is here rather than on the relationships themselves:
            # extract.log_failed_relationships rejects any relationship object
            # carrying a third key, so annotating the edge would make the
            # record fail its own validator.
            doc.setdefault("provenance", {})["grandparent_sides"] = {
                "source": "backfill_grandparent_sides.py",
                "authority": "Daniel Genkins, 2026-08-10",
                "resolved_from": "the entry's own transcription",
                **{k: stats[k] for k in ("seen", "maternal", "paternal",
                                         "unresolved")},
                "unresolved_by_reason": {r: stats[r] for r in REASONS if stats[r]},
            }
            payload = doc
        else:
            payload = labelled
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if audit:
            base = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(out_dir, f"{base}.unsided.json"),
                      "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False, indent=2)

    return stats, shown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+",
                    help="assembled volume json files (globs allowed)")
    ap.add_argument("--out", default=None,
                    help="directory to write relabelled volumes to. Omit for a "
                         "report only -- nothing is ever rewritten in place.")
    ap.add_argument("--samples", type=int, default=0,
                    help="print up to N entries whose grandparents could not be "
                         "sided, with the side clauses found in each")
    args = ap.parse_args(argv)

    paths = [p for pat in args.paths for p in sorted(glob.glob(pat))] or args.paths
    total = collections.Counter()
    print(f"{'volume':>22} {'seen':>6} {'maternal':>9} {'paternal':>9} "
          f"{'unsided':>8} {'resolved':>9}")
    for path in paths:
        stats, shown = process(path, args.out, args.samples)
        total.update(stats)
        seen = stats["seen"]
        rate = (stats["maternal"] + stats["paternal"]) / seen if seen else 0.0
        print(f"{os.path.basename(path)[:22]:>22} {seen:6d} {stats['maternal']:9d} "
              f"{stats['paternal']:9d} {stats['unresolved']:8d} {100 * rate:8.1f}%")
        for eid, sides in shown:
            print(f"    unresolved in {eid}: side clauses = "
                  f"{[s.split()[0] for s in sides] or 'none'}")

    seen = total["seen"]
    rate = (total["maternal"] + total["paternal"]) / seen if seen else 0.0
    print(f"{'TOTAL':>22} {seen:6d} {total['maternal']:9d} {total['paternal']:9d} "
          f"{total['unresolved']:8d} {100 * rate:8.1f}%")
    print(f"\n{total['entries_with_grandparents']} entries carry a grandparent "
          f"edge; {total['entries_with_unresolved']} still hold at least one "
          f"unsided grandparent.")

    # The split Daniel needs. The first two lines are the register's own
    # silence; the last is almost always an extraction defect -- an edge
    # pointing at someone the side clauses never name, usually the godparent --
    # so it is a quality signal, not a coverage shortfall.
    if total["unresolved"]:
        blurb = {
            "no_side_clause": "record never says which side",
            "no_name": "grandparent has no usable name",
            "name_in_both_clauses": "same name on both sides, undecidable",
            "name_in_no_clause": "named in NO side clause -- likely an "
                                 "extraction defect, worth reviewing",
        }
        print("\nwhy the rest stayed unsided:")
        for reason in REASONS:
            if total[reason]:
                print(f"  {total[reason]:5d}  {reason:<21} {blurb[reason]}")

    if args.out:
        print(f"\nwritten to {args.out}")
        print("  <volume>.unsided.json lists every unsided grandparent by "
              "entry, name and reason.")
    else:
        print("\nreport only -- pass --out DIR to write the relabelled volumes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
