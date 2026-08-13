"""How bad is the latent co-participant defect? Measure it, today, for free.

THE LATENT DEFECT
-----------------
HANDOVER sec.14: the same-entry veto is doing the heavy lifting against
co-participant false merges, and "the moment the corpus contains two registers
covering the same parish, co-participants will start appearing in different
entries, and nothing will stop them merging."

That has never been measured, on the stated grounds that the situation does not
exist in these 7 volumes -- they come from 7 different institutions.

IT DOES EXIST. Entries sharing an identical extracted payload ARE one event
recorded twice in two different entries. Every cross-copy pair between them is exactly
the configuration sec.14 warns about:

    - same event, same date, same parish, same associates
    - DIFFERENT entry ids, so the same-entry veto never fires
    - and for i != j, guaranteed DIFFERENT people

So the latent failure is observable now, without waiting for the archive to
supply a same-parish register pair. This is the same trick as the rest of the
free supervision: the corpus already contains the label.

WHAT IS MEASURED
----------------
Across the two copies of a duplicated entry:

    SAME person   (local id i vs i)  -> should merge. Recall on the easy case.
    DIFFERENT     (local id i vs j)  -> must NOT merge. This is the latent bug.

Run against `production/luna_v3/assembled` -- the RAW assembly. Pointing it at
`assembled_deduped` measures nothing, because de-duplication is precisely what
removes the second copy. The script refuses that corpus rather than silently
reporting zeros, which is this project's most common way of producing a clean
wrong answer.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, score
from ssda_nlp_tools.volume_geo import load as load_geo


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if "dedup" in a.assembled:
        raise SystemExit(
            "refusing: the de-duplicated corpus has had the second copy of "
            "every duplicated entry removed, so this measurement would report "
            "zero pairs and read as 'no problem'. Use the raw assembly.")

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {a.assembled!r}")
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    by_entry = collections.defaultdict(list)
    for m in M:
        by_entry[m["_entry"]].append(m)
    entry_by_id = {e.get("id"): e for e in entries}

    # Duplicate entries: identical extracted payload, different entry id.
    # Empty payloads all hash alike and are excluded -- conflating them with
    # real duplicates is the 380-vs-59 trap from HANDOVER sec.3.
    # GROUPED ON THE FULL EXTRACTED PAYLOAD -- people AND events -- which is
    # exactly what dedupe_entries.py's payload_hash does.
    #
    # An earlier version keyed on PEOPLE ONLY, on the theory that differing
    # transcription text over identical people meant one event written twice.
    # Measured, that is false: of 62 people-only groups, 28 hold genuinely
    # DIFFERENT events -- burials four years apart, and two adjacent entries
    # recording burials two days apart. Identical people lists arise because the
    # extraction is sparse, not because the record is a copy. Collapsing them
    # would destroy real distinct burials, and the "extend dedupe to payload
    # identity" recommendation built on that grouping was wrong.
    groups = collections.defaultdict(list)
    for eid, people in sorted(by_entry.items()):
        if not people:
            continue                       # empty payloads all hash alike
        e = entry_by_id.get(eid) or {}
        key = json.dumps(e.get("data") or {}, sort_keys=True, ensure_ascii=False)
        groups[key].append(eid)
    dup_groups = [v for v in groups.values() if len(v) > 1]

    print(f"corpus            : {len(entries):,} entries, {len(M):,} mentions")
    print(f"duplicate groups  : {len(dup_groups)} groups covering "
          f"{sum(len(v) for v in dup_groups)} entries "
          f"(sizes {dict(collections.Counter(len(v) for v in dup_groups))})")
    print(f"                    grouped on the full data payload, as dedupe does\n")
    if not dup_groups:
        raise SystemExit("no duplicate entries found -- nothing to measure")

    # UNIT: one distinct pair of PEOPLE, counted once.
    #
    # Two bugs lived here and both inflated the answer, which is why the count
    # is computed this way and not the obvious way:
    #
    #   1. Iterating A x B visits (i,j) and (j,i), double-counting every pair.
    #   2. A group of 3+ copies yields several copy-pairs of the SAME two
    #      people, so counting per copy-pair counts one confusion repeatedly.
    #
    # Together they reported 130 where the honest figure is 50. HANDOVER sec.9
    # rule 4: pairs are not people. State the unit.
    same, diff = [], []
    for eids in dup_groups:
        ea, eb = eids[0], eids[1]          # one representative copy-pair
        A = {m["_local_id"]: m for m in by_entry[ea]}
        B = {m["_local_id"]: m for m in by_entry[eb]}
        shared_ids = sorted(set(A) & set(B))
        for i, ia in enumerate(shared_ids):
            r = score(A[ia], B[ia], stats, geo=geo, vol_of=vol_of)
            same.append((r["log_odds"], r.get("vetoed"), A[ia], B[ia]))
            for ib in shared_ids[i + 1:]:
                x, y = A[ia], B[ib]
                r = score(x, y, stats, geo=geo, vol_of=vol_of)
                diff.append((r["log_odds"], r.get("vetoed"), x, y))

    def report(label, rows, want_merge):
        n = len(rows)
        if not n:
            print(f"{label}: none")
            return {}
        vetoed = sum(1 for lo, v, _, _ in rows if v)
        scored = [lo for lo, v, _, _ in rows if not v]
        auto = sum(1 for lo in scored if lo >= E.AUTO_MERGE_LOG_ODDS)
        rev = sum(1 for lo in scored if E.REVIEW_LOG_ODDS <= lo < E.AUTO_MERGE_LOG_ODDS)
        s = sorted(scored)
        med = s[len(s) // 2] if s else float("nan")
        print(f"{label}")
        print(f"    pairs                 : {n:,}")
        print(f"    refused by a veto     : {vetoed:,}")
        print(f"    scored                : {len(scored):,}   median log-odds "
              f"{med:+.2f}")
        verdict = "MISSED MERGES" if want_merge else "FALSE MERGES  <-- the defect"
        print(f"    would AUTO-MERGE      : {auto:,}"
              f"   ({100*auto/max(1,len(scored)):.1f}% of scored)")
        print(f"    would go to review    : {rev:,}")
        if not want_merge and auto:
            print(f"    {verdict}: {auto:,}")
        elif want_merge:
            print(f"    recall on the easy case: "
                  f"{100*auto/max(1,len(scored)):.1f}% auto, "
                  f"{100*(auto+rev)/max(1,len(scored)):.1f}% at least review")
        return {"pairs": n, "vetoed": vetoed, "scored": len(scored),
                "auto": auto, "review": rev, "median_log_odds": med}

    out = {"corpus": a.assembled,
           "duplicate_groups": len(dup_groups),
           "entries_in_groups": sum(len(v) for v in dup_groups),
           "grouping": "extracted people payload, not byte-identity"}
    print("SAME PERSON across the two copies (should merge)")
    out["same"] = report("", same, want_merge=True)
    print()
    print("DIFFERENT PEOPLE across the two copies -- co-participants in ONE")
    print("event, in DIFFERENT entries, so the same-entry veto cannot fire.")
    print("This is the latent defect of HANDOVER sec.14, made observable.")
    out["different"] = report("", diff, want_merge=False)

    if out["different"].get("auto"):
        merged = [r for r in diff
                  if not r[1] and r[0] >= E.AUTO_MERGE_LOG_ODDS]
        idn = [r for r in merged
               if str(r[2].get("name") or "").strip().lower()
               == str(r[3].get("name") or "").strip().lower()]
        print(f"\n  of those false merges, the two names are IDENTICAL in "
              f"{len(idn)} and differ in {len(merged) - len(idn)}")
        print("  Identical-name pairs are the WEAK ones: the premise that two")
        print("  local ids are two people is the extractor's separation, and a")
        print("  namesake split in two would look exactly like this. The")
        print("  different-name pairs are the defensible core.")
        out["different"]["auto_identical_names"] = len(idn)
        out["different"]["auto_different_names"] = len(merged) - len(idn)

        print("\nWORST FALSE MERGES, DIFFERENT NAMES ONLY")
        print("(read them; HANDOVER sec.9 rule 9 -- aggregates hide role bugs)")
        worst = sorted((r for r in merged if r not in idn),
                       key=lambda r: -r[0])[:10]
        for lo, _, x, y in worst:
            print(f"  {lo:+6.2f}  {str(x.get('name'))[:32]:<32} vs "
                  f"{str(y.get('name'))[:32]}")

    print("\nSCOPE. These are duplicate transcriptions of one entry, so the two")
    print("copies agree on EVERYTHING. A real same-parish register pair would")
    print("agree less, so treat this as the worst case for that scenario --")
    print("an upper bound on the latent rate, not an estimate of it.")

    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
