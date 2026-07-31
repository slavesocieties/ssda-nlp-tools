#!/usr/bin/env python3
"""withdraw_records.py — remove records whose text is not the manuscript.

Offline, $0, no network, no key.

    python withdraw_records.py --dry-run
    python withdraw_records.py --confirm

Daniel, 2026-07-31: "withdraw those with model apology text."

Two records in the delivered corpus contain the transcription model's own words
instead of the register's. 201991-0304-A-05 reads, in full, "En la Yglesia
Parroquial de Ntra. Senora de la Asuncion I cannot fulfill this request. I am
programmed to be a helpful and harmless AI assistant." The other, 201991-0275-A-04,
runs normally for a line and a half and then breaks into an apology.

WITHDRAWN, NOT DELETED
----------------------
Every withdrawn record is written whole to a quarantine file first, with the
reason and the detector output that found it. Three reasons that matter here:

  - This is a scholarly database. "Record 201991-0304-A-05 was removed on
    2026-07-31 because its transcription was fabricated" is a fact someone may
    need in a year, and a record that simply vanishes cannot be audited.
  - The entry ids stay reserved. Nothing renumbers, so any external reference to
    a withdrawn id still resolves to an explanation rather than to a different
    person's baptism.
  - It is reversible. If a page is re-transcribed successfully the record can be
    restored under its original id.

The corpus `coverage` block is updated to match, because a file claiming 2,021
records while holding 2,019 is worse than either number alone.
"""
import argparse
import json
import os
import shutil
from datetime import date

from ssda_nlp_tools.transcription_integrity import check_page

DEFAULT_REASON = ("transcription contains the model's own refusal text rather "
                  "than the manuscript; the record is fabricated, not merely "
                  "inaccurate")


def load_plan(path):
    d = json.load(open(path, encoding="utf-8"))
    return list(d.get("entry_ids_retranscribe") or [])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--plan", default="production/repair_20260731/bad_records.json")
    ap.add_argument("--quarantine",
                    default="production/luna_v3/withdrawn_records.json")
    ap.add_argument("--ids", nargs="*", help="override the plan's id list")
    ap.add_argument("--confirm", action="store_true",
                    help="actually rewrite the corpus files")
    args = ap.parse_args(argv)

    targets = set(args.ids or load_plan(args.plan))
    if not targets:
        ap.error("no ids to withdraw")
    print(f"withdrawing {len(targets)} record(s): {sorted(targets)}\n")

    withdrawn, touched = [], {}
    for name in sorted(os.listdir(args.assembled)):
        if not name.endswith(".materialized.json"):
            continue
        path = os.path.join(args.assembled, name)
        d = json.load(open(path, encoding="utf-8"))
        entries = d.get("entries") or []
        keep, gone = [], []
        for e in entries:
            if str(e.get("id")) in targets:
                gone.append(e)
            else:
                keep.append(e)
        if not gone:
            continue

        for e in gone:
            # Re-run the detector rather than trusting the plan file, so the
            # quarantine record carries evidence gathered at withdrawal time.
            chk = check_page(e.get("text_faithful") or e.get("normalized") or "",
                             str(e.get("id")))
            withdrawn.append({
                "id": e.get("id"),
                "volume": d.get("volume"),
                "withdrawn_on": date.today().isoformat(),
                "authorised_by": "Daniel Genkins, 2026-07-31",
                "reason": DEFAULT_REASON,
                "detector": chk,
                "restorable": True,
                "record": e,
            })
            flagged = "flagged" if not chk["ok"] else "NOT FLAGGED"
            print(f"  {e.get('id')}  ({d.get('volume')})  detector: {flagged} "
                  f"{chk['codes']}")
            if chk["ok"]:
                print("     ^ the integrity gate does not flag this record. "
                      "Withdrawing anyway because the id was named, but check "
                      "the id before confirming.")

        d["entries"] = keep
        cov = d.get("coverage") or {}
        cov["materialized_records"] = len(keep)
        cov["withdrawn_records"] = cov.get("withdrawn_records", 0) + len(gone)
        cov["withdrawn_note"] = (f"{cov['withdrawn_records']} record(s) withdrawn; "
                                 f"see {os.path.basename(args.quarantine)}")
        d["coverage"] = cov
        touched[path] = d
        print(f"    {name}: {len(entries)} -> {len(keep)} entries")

    missing = targets - {w["id"] for w in withdrawn}
    if missing:
        print(f"\n!! not found in the corpus: {sorted(missing)}")
        print("   (already withdrawn, or the id is wrong -- resolve before confirming)")

    if not args.confirm:
        print("\nDRY RUN. Nothing written. Re-run with --confirm.")
        return 0

    os.makedirs(os.path.dirname(args.quarantine) or ".", exist_ok=True)
    existing = []
    if os.path.exists(args.quarantine):
        existing = json.load(open(args.quarantine, encoding="utf-8")).get("withdrawn", [])
    already = {w["id"] for w in existing}
    merged = existing + [w for w in withdrawn if w["id"] not in already]
    with open(args.quarantine, "w", encoding="utf-8") as f:
        json.dump({"withdrawn": merged}, f, ensure_ascii=False, indent=1)
    print(f"\n-> quarantined {len(withdrawn)} record(s) to {args.quarantine}")

    for path, d in touched.items():
        shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        print(f"-> rewrote {path}  (previous kept as .bak)")
    print("\nRe-run assemble/merge downstream artifacts; the corpus has changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
