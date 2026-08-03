#!/usr/bin/env python3
"""find_bad_records.py — records that need re-doing, and why.

Offline, $0, no network, no key.

Two failure classes, found while checking the composition of 201991 rather than
by any existing test. Both are small; both are the kind that survive every
automated check we have, because the output is well formed.

  REFUSAL   The transcription model's own apology embedded in the faithful text
            and delivered as manuscript content:

              "En la Yglesia Parroquial de Ntra. Senora de la Asuncion
               I cannot fulfill this request. I am programmed to be a helpful
               and harmless AI assistant."

            This is fabricated text in a historical record. It is the most
            serious of the two regardless of how few there are, and the count
            is NOT the point -- one is too many for a scholarly database.

            Note the phrasing is why we missed it. The bake-off already looks
            for embedded API failures, but it looks for "transcription error"
            and "unable to process", not for a first-person apology.

  NO_EVENT  A long record whose text plainly describes a sacrament ("cadaver",
            "murio", "contrajo matrimonio") but which carries no event. Short
            event-less records are margin annotations and are correct; the
            filter is length plus an explicit verb, so it does not flag those.

Neither class is repaired here. Inventing an event we did not extract, or
paraphrasing a page we could not read, would put made-up content into the
corpus -- which is the exact failure being reported. This stages the work and
reports it.

    python find_bad_records.py
    python find_bad_records.py --stage production/repair_20260731
"""
import argparse
import glob
import json
import os
import re

# First-person model apologies. Deliberately broad: a false positive costs a
# human ten seconds, a false negative leaves fabricated text in the archive.
REFUSAL = re.compile(
    r"(I'?m sorry|I cannot|I can'?t|I am unable|I'?m not able|I apologize"
    r"|as an AI|language model|cannot (?:transcribe|fulfill|assist|provide)"
    r"|unable to (?:transcribe|process|read)|helpful and harmless"
    r"|no puedo (?:transcribir|ayudar)|lo siento|nao posso)", re.I)

# Explicit sacramental verbs. Not a general word list: each of these states that
# the act happened, so a record containing one and no event is a miss.
SACRAMENT = re.compile(
    r"\b(muri[oó]|falleci[oó]|fall?eceu|sepult|enterr|cad[aá]ver|cadaver"
    r"|baut[ií][csz]|batiz|casad[oa] y velad|despos|contrajo matrimonio"
    r"|recebe?r[aã]o em matrim)\b", re.I)

MIN_CHARS = 400          # below this an event-less record is margin annotation


def scan(paths):
    refusal, no_event, short_none = [], [], 0
    total = 0
    for path in paths:
        vol = os.path.basename(path).split(".")[0]
        for e in json.load(open(path, encoding="utf-8")).get("entries") or []:
            total += 1
            faithful = e.get("text_faithful") or ""
            for field in ("text_faithful", "normalized"):
                m = REFUSAL.search(e.get(field) or "")
                if m:
                    refusal.append({"volume": vol, "id": e["id"], "field": field,
                                    "matched": m.group(0),
                                    "context": (e.get(field) or "")[
                                        max(0, m.start() - 60):m.start() + 100]})
                    break
            if ((e.get("data") or {}).get("events") or []):
                continue
            if len(faithful) < MIN_CHARS:
                short_none += 1          # margin annotation, correctly no event
                continue
            m = SACRAMENT.search(faithful)
            if m:
                no_event.append({"volume": vol, "id": e["id"],
                                 "chars": len(faithful), "verb": m.group(0),
                                 "text": faithful[:200]})
    return refusal, no_event, short_none, total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--stage", metavar="DIR",
                    help="write the affected entry ids for a targeted re-run")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json")))
    refusal, no_event, short_none, total = scan(paths)

    print(f"scanned {total:,} delivered records\n")
    print(f"REFUSAL   model apology delivered as manuscript text : {len(refusal)}")
    for r in refusal:
        print(f"    [{r['id']}] {r['field']} matched {r['matched']!r}")
        print(f"        ...{r['context'].strip()}...")
    print(f"\nNO_EVENT  long record, explicit sacrament, no event  : {len(no_event)}")
    for r in no_event:
        print(f"    [{r['id']}] {r['chars']} chars, {r['verb']!r}")
    print(f"\n          (short event-less records, correctly margin "
          f"annotations: {short_none:,})")

    # A record can be in BOTH lists, and one is: 201991-0275-A-04 has no event
    # BECAUSE the refusal truncated its text mid-sentence. Summing the two
    # classes double-counts it and overstates the damage.
    both = {r["id"] for r in refusal} & {r["id"] for r in no_event}
    distinct = len({r["id"] for r in refusal} | {r["id"] for r in no_event})
    print(f"\naffected {distinct} DISTINCT records of {total:,} "
          f"({100*distinct/total:.3f}%)")
    if both:
        print(f"  {len(both)} in both classes: {sorted(both)}")
        print("  these have no event BECAUSE the refusal cut the text short, so")
        print("  re-extraction cannot fix them -- they need re-transcription.")
    print("\nNothing is repaired here. Inventing an event we did not extract, or")
    print("paraphrasing a page we could not read, would put made-up content into")
    print("the corpus -- which is the failure being reported.")

    if args.stage:
        os.makedirs(args.stage, exist_ok=True)
        # re-extraction cannot fix a record whose SOURCE TEXT is truncated
        reextract = [r["id"] for r in no_event if r["id"] not in both]
        plan = {"refusal": refusal, "no_event": no_event,
                "in_both": sorted(both),
                "note": "REFUSAL needs RE-TRANSCRIPTION (the source text is "
                        "wrong). NO_EVENT needs RE-EXTRACTION only (the source "
                        "text is fine; the extractor missed the event).",
                "entry_ids_retranscribe": [r["id"] for r in refusal],
                "entry_ids_reextract": reextract}
        out = os.path.join(args.stage, "bad_records.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"\n-> {out}")
        print("   the two classes need DIFFERENT fixes: refusals are a "
              "transcription problem, missed events an extraction one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
