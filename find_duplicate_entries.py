"""Count duplicate transcriptions that the hard vetoes can never merge.

Daniel, 2026-08-07, on why real examples make poor training data: transcription
issues -- "hallucinated duplicates, same-sex couples" -- yield "useless or
actively counter-productive training data."

Two of the pipeline's vetoes turn that upstream noise into permanent error:

  same-entry                two people extracted from ONE entry can never be
                            merged, so a person emitted twice inside an entry is
                            two people forever.
  both-sacrament-principals you are baptised once, so two baptism principals in
                            DIFFERENT entries can never be merged -- including
                            when the two entries are the same baptism
                            transcribed twice.

Both vetoes are correct as stated. The problem is that they are stated about
PEOPLE while being applied to RECORDS, and a duplicated record is not a second
person. This counts how often that happens, so the cost of the current design is
a number rather than an anecdote.

A pair is counted as a near-certain duplicate only when it agrees on everything
a scribe would have had to repeat: same volume, same year, same normalized name,
and an IDENTICAL non-empty relationship context. That is deliberately strict --
it will undercount -- because an inflated figure here would be worse than none.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import ssda_nlp_tools.disambiguate as D
from ssda_nlp_tools.textmatch import normalize_name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    print(f"{len(entries):,} entries, {len(M):,} mentions\n")

    # --- 1. duplicates INSIDE one entry (killed by the same-entry veto) -------
    within = collections.defaultdict(list)
    for m in M:
        within[(m["_entry"], normalize_name(m.get("name")),
                frozenset(m.get("_ctx") or ()))].append(m)
    dup_within = {k: v for k, v in within.items() if len(v) > 1}
    n_within = sum(len(v) - 1 for v in dup_within.values())
    print(f"A. SAME PERSON EMITTED TWICE INSIDE ONE ENTRY")
    print(f"   {len(dup_within):,} distinct duplications, {n_within:,} surplus mentions")
    print(f"   blocked by: the same-entry veto -- unmergeable by construction")
    for k, v in list(dup_within.items())[:a.show]:
        print(f"     {k[0]}  {v[0].get('name')!r} x{len(v)}  ids="
              f"{[m['_local_id'] for m in v]}")

    # --- 2. duplicate ENTRIES (killed by both-sacrament-principals) -----------
    key = collections.defaultdict(list)
    for m in M:
        if not m.get("_unique_sacrament"):
            continue
        ctx = frozenset(m.get("_ctx") or ())
        if not ctx:
            continue                       # no evidence to call it a duplicate
        vol = str(m["_entry"]).split("-")[0]
        key[(vol, m.get("_year"), normalize_name(m.get("name")), ctx)].append(m)

    groups = {k: v for k, v in key.items()
              if len({m["_entry"] for m in v}) > 1}
    n_pairs = sum(len({m["_entry"] for m in v}) - 1 for v in groups.values())
    print(f"\nB. ONE SACRAMENT TRANSCRIBED INTO SEVERAL ENTRIES")
    print(f"   {len(groups):,} sacraments appear in >1 entry, "
          f"{n_pairs:,} surplus identities")
    print(f"   blocked by: the both-sacrament-principals veto -- also "
          f"unmergeable by construction")
    shown = 0
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if shown >= a.show:
            break
        shown += 1
        vol, year, name, ctx = k
        ents = sorted({m["_entry"] for m in v})
        print(f"     vol {vol} y={year} {name!r} in {len(ents)} entries: "
              f"{', '.join(ents[:4])}{' ...' if len(ents) > 4 else ''}")
        print(f"        shared context: "
              f"{', '.join(f'{r}={n}' for r, n in sorted(ctx))[:100]}")

    total = n_within + n_pairs
    print(f"\n  TOTAL surplus identities from duplicated records: {total:,}")
    print(f"  as a share of the 33,250 identities in the current run: "
          f"{100*total/33250:.2f}%")
    print("""
  These are a FLOOR, not an estimate. The test requires an identical
  relationship context, so any duplicate whose second transcription differs by
  one godparent name is not counted here.

  Note what this does NOT say: it does not say the vetoes are wrong. Merging two
  baptisms is the error they exist to prevent. It says the vetoes are applied to
  records as if records were people, so de-duplication has to happen BEFORE
  disambiguation rather than inside it.""")


if __name__ == "__main__":
    main()
