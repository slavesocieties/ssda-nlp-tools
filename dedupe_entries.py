"""Collapse duplicated records BEFORE disambiguation, where they can still be fixed.

THE PROBLEM
-----------
Two of the merge stage's vetoes are correct about people and wrong about records:

  same-entry                 two people extracted from ONE entry can never merge
  both-sacrament-principals  you are baptised once, so two baptism principals in
                             different entries can never merge

Neither is wrong as stated. But a record transcribed twice is not a second
person, and both vetoes are absolute, so a duplicated record becomes a permanent
phantom identity that no amount of evidence can undo. Daniel graded one such
pair 75 (`201991-0251-B-01` / `-B-02`, the same 1881 baptism of Josefa); the
scorer returns -inf on it.

De-duplication therefore has to happen here, before the vetoes apply.

THE RULE, AND WHY IT IS THIS CONSERVATIVE
-----------------------------------------
Only records whose ENTIRE data payload is byte-identical, within one volume, are
collapsed. Near-duplicates that differ by a single godparent name are NOT
touched: telling a duplicated record from two genuinely similar baptisms in the
same parish is exactly the judgement this pipeline exists to make, and making it
here -- irreversibly, before anything can weigh evidence -- would be the wrong
place. This will undercount. That is the intended failure direction.

THE SAFETY RAIL THAT MATTERS MOST
---------------------------------
**An empty payload is not an identity.** 334 of 6,794 entries (4.9%) have no
extracted people at all, because their extraction failed -- and they all hash
identically. A de-duplicator that grouped by hash alone would silently collapse
them into a handful of entries and destroy 333 real records. Empty payloads are
counted and reported, never merged.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os


def load_redirects(report="production/luna_v3/dedupe_report.json"):
    """{dropped entry id -> the identical entry that survived}.

    De-duplication REMOVES entries, and a label set graded before it points at
    entry ids that no longer exist. Promoting the de-duplicated corpus without
    this orphaned 8 of the 300 pairs Daniel graded -- silently, because a
    resolver that finds nothing just returns fewer rows.

    The redirect is exact rather than approximate: entries are collapsed only
    when their whole payload is byte-identical, so the surviving entry contains
    the same people with the same local ids. A label pointing at the dropped
    copy and one pointing at the survivor are about the same person.
    """
    try:
        rep = json.load(open(report, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {r["dropped"]: r["kept"] for r in rep.get("collapsed_entries", [])}


def check_redirects(redirects, corpus_entry_ids):
    """Do these redirects actually describe THIS corpus? Returns a list of faults.

    The report is a second source of truth about which entries exist, and a stale
    one repoints label resolution silently: every graded pair still resolves, just
    to the wrong mention. Nothing downstream can notice, because a redirect that
    lands on a real entry looks exactly like a redirect that lands on the right
    one.

    Two invariants have to hold if the report was produced from this corpus:

        every `kept` id EXISTS   -- it survived de-duplication
        every `dropped` id is GONE -- it was collapsed away

    Checking them needs no fingerprint stored in the file, so it works on the
    report as it stands and cannot itself require regenerating a frozen artifact.
    """
    ids = set(corpus_entry_ids)
    faults = []
    missing_kept = sorted(k for k in set(redirects.values()) if k not in ids)
    present_dropped = sorted(d for d in redirects if d in ids)
    if missing_kept:
        faults.append(
            f"{len(missing_kept)} 'kept' entries are absent from the corpus "
            f"(e.g. {missing_kept[:3]}) -- the report describes a different corpus")
    if present_dropped:
        faults.append(
            f"{len(present_dropped)} 'dropped' entries are still present "
            f"(e.g. {present_dropped[:3]}) -- the corpus was not de-duplicated "
            f"with this report")
    return faults


def resolve_entry(entry_id, redirects):
    """Follow a dropped entry to its surviving twin, if it was collapsed."""
    seen = set()
    while entry_id in redirects and entry_id not in seen:
        seen.add(entry_id)
        entry_id = redirects[entry_id]
    return entry_id


def payload_hash(entry):
    return hashlib.sha1(
        json.dumps(entry.get("data") or {}, sort_keys=True,
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def n_people(entry):
    return len((entry.get("data") or {}).get("people") or [])


def entry_id(entry):
    return str(entry.get("entry") or entry.get("id") or "")


def dedupe_people_within(entry):
    """Collapse a person emitted twice inside ONE entry.

    `701157-0214-01` lists "Fernando Jose da Costa" twice under the same id
    `P01`. Because same-entry is a veto, that is a phantom identity forever.

    The key is the ID ALONE, not the whole record. A person id is the entry's own
    primary key -- two people sharing one is an extraction error by definition,
    whatever else differs. Requiring the rest to match too found NOTHING here,
    because the duplicate Fernando carries a spouse relationship the copy lacks;
    duplicates are usually PARTIAL, which is exactly why an exact-match rule
    misses them.

    Relationships are UNIONED rather than dropped, so collapsing never loses an
    edge. If two people share an id but carry genuinely different names, that is
    a different error and is reported instead of merged -- picking one would
    silently discard a person.
    """
    people = (entry.get("data") or {}).get("people") or []
    by_id, order, conflicts = {}, [], []
    for p in people:
        pid = str(p.get("id"))
        if pid not in by_id:
            by_id[pid] = dict(p)
            order.append(pid)
            continue
        kept = by_id[pid]
        a, b = str(kept.get("name") or "").strip(), str(p.get("name") or "").strip()
        if a and b and a.casefold() != b.casefold():
            conflicts.append({"id": pid, "names": [a, b]})
            order.append(pid + "\x00dup")          # keep both, untouched
            by_id[pid + "\x00dup"] = dict(p)
            continue
        if not kept.get("name") and b:
            kept["name"] = p.get("name")
        seen = {json.dumps(r, sort_keys=True, ensure_ascii=False)
                for r in (kept.get("relationships") or [])}
        for r in (p.get("relationships") or []):
            if json.dumps(r, sort_keys=True, ensure_ascii=False) not in seen:
                kept.setdefault("relationships", []).append(r)
        kept["_merged_duplicate"] = True
    removed = [pid for pid in {p_id for p_id in
                               (str(x.get("id")) for x in people)}
               if sum(1 for x in people if str(x.get("id")) == pid) > 1]
    keep = [by_id[k] for k in order]
    return keep, sorted(removed), conflicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--outdir", default="production/luna_v3/assembled_deduped")
    ap.add_argument("--report", default="production/luna_v3/dedupe_report.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.assembled, "*.materialized.json")))
    if not paths:
        raise SystemExit(f"no *.materialized.json under {a.assembled!r} -- "
                         f"refusing to write an empty corpus over anything")

    report = {"source": a.assembled, "volumes": [], "collapsed_entries": [],
              "collapsed_people": [], "empty_entries": [], "id_conflicts": []}
    tot_in = tot_out = tot_empty = tot_people_removed = 0

    for path in paths:
        doc = json.load(open(path, encoding="utf-8"))
        entries = doc.get("entries") or []
        tot_in += len(entries)

        # --- within-entry duplicate people -------------------------------
        for e in entries:
            keep, removed, conflicts = dedupe_people_within(e)
            if conflicts:
                report["id_conflicts"].append(
                    {"entry": entry_id(e), "conflicts": conflicts})
            if removed:
                before = len((e.get("data") or {}).get("people") or [])
                e["data"]["people"] = keep
                tot_people_removed += before - len(keep)
                report["collapsed_people"].append(
                    {"entry": entry_id(e), "duplicate_ids": removed})

        # --- cross-entry identical records --------------------------------
        first_by_hash, out = {}, []
        for e in entries:
            if n_people(e) == 0:
                # NOT a duplicate: a failed extraction. Kept, and counted.
                tot_empty += 1
                report["empty_entries"].append(entry_id(e))
                out.append(e)
                continue
            h = payload_hash(e)
            if h in first_by_hash:
                kept = first_by_hash[h]
                prov = kept.setdefault("duplicate_of", [])
                prov.append(entry_id(e))
                report["collapsed_entries"].append(
                    {"kept": entry_id(kept), "dropped": entry_id(e)})
                continue
            first_by_hash[h] = e
            out.append(e)

        tot_out += len(out)
        vol = os.path.basename(path).split(".")[0]
        report["volumes"].append({"volume": vol, "in": len(entries),
                                  "out": len(out)})
        if not a.dry_run:
            os.makedirs(a.outdir, exist_ok=True)
            doc["entries"] = out
            json.dump(doc, open(os.path.join(a.outdir, os.path.basename(path)),
                                "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)

    print(f"{len(paths)} volumes, {tot_in:,} entries in -> {tot_out:,} out")
    print(f"  duplicate records collapsed  : {len(report['collapsed_entries']):,}")
    print(f"  duplicate people within entry: {tot_people_removed:,}")
    print(f"  empty payloads KEPT, not merged: {tot_empty:,} "
          f"({100*tot_empty/tot_in:.2f}% of the corpus)")
    if tot_in - tot_out != len(report["collapsed_entries"]):
        raise SystemExit("BUG: entries lost that were not recorded as collapsed")
    print("  accounting checks out: every dropped entry is recorded\n")

    print(f"{'volume':>10} {'in':>7} {'out':>7} {'dropped':>8}")
    for v in report["volumes"]:
        print(f"{v['volume']:>10} {v['in']:7,} {v['out']:7,} "
              f"{v['in']-v['out']:8,}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    json.dump(report, open(a.report, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n-> {a.outdir}\n-> {a.report}")
    print("\nThe kept copy of each duplicated record carries a `duplicate_of` list,")
    print("so the collapse is traceable back to the original entry ids.")


if __name__ == "__main__":
    main()
