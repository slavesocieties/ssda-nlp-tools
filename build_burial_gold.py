#!/usr/bin/env python3
"""build_burial_gold.py — candidate burial examples for Daniel to correct into gold.

Offline, $0, no network, no key.

Why this exists. Daniel's gold (`training_data.json`, 15 examples) contains 11
baptisms, 9 births and 4 marriages, and **zero burials**. The delivered corpus
contains 2,169 burial events, more than any other type, and 201991 -- the
largest volume at 2,021 records -- is largely a burial register. So the accuracy
figures we have reported are measured on a gold set that does not include the
single commonest thing the extractor is asked to do. That is a measurement gap,
not a known-good result.

Design choice worth stating: the candidates are **pre-filled with the current
extraction** rather than left blank. Authoring 12 gold entries from raw text is
hours of a historian's time; correcting 12 drafts is not. The cost is anchoring
-- a reviewer nudged toward agreeing with the model -- so every file is labelled
as model output pending correction, never as gold, and the raw text is placed
above the draft so it can be read first.

Selection maximises the variety of things that can go wrong rather than sampling
at random, because 12 random burials from a 2,169-entry register would be twelve
near-copies of the same formula.

    python build_burial_gold.py --n 12
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

SPANISH = re.compile(r"\b(se enterr|sepultura|cementerio|di[oó] sepultura|"
                     r"cad[aá]ver|falleci)", re.I)
PORTUGUESE = re.compile(r"\b(faleceu|sepultado|foi enterrado|encomendou|"
                        r"sepultura ecclesi)", re.I)


def language_of(text: str) -> str:
    """Cheap and honest: count register-specific burial formulae. Daniel can fix
    the field in two seconds if a volume is mixed."""
    pt = len(PORTUGUESE.findall(text))
    es = len(SPANISH.findall(text))
    if pt > es:
        return "Portuguese"
    return "Spanish"


def burial_entries(paths):
    for p in paths:
        vol = os.path.basename(p).split(".")[0]
        for e in json.load(open(p, encoding="utf-8")).get("entries") or []:
            data = e.get("data") or {}
            events = data.get("events") or []
            if not any(str(ev.get("type", "")).lower() == "burial" for ev in events):
                continue
            yield vol, e, data, events


def decade_of(events):
    """From the EXTRACTED event dates, not from the text.

    These registers spell years out ("mil ochocientos cuarenta"), so scanning the
    transcription for 4-digit years finds almost nothing real -- the few hits are
    page numbers and sums of money, which is worse than no signal because it
    silently mis-buckets entries.
    """
    years = []
    for ev in events or []:
        m = re.match(r"\s*(\d{4})", str(ev.get("date") or ""))
        if m:
            years.append(int(m.group(1)))
    return (min(years) // 10 * 10) if years else 0


def profile(e, data, events):
    """The axes along which a burial entry can be hard. Used as a diversity key,
    so the sample spans failure modes instead of spanning the formula."""
    people = data.get("people") or []
    rels = [r for p in people for r in (p.get("relationships") or [])]
    text = e.get("text_faithful") or e.get("normalized") or ""
    principals = {str(pid) for ev in events
                  if str(ev.get("type", "")).lower() == "burial"
                  for pid in (ev.get("principals") or [])}
    named = [p for p in people if (p.get("name") or "").strip()]
    return {
        "lang": language_of(text),
        "decade": decade_of(events),
        "n_people": min(len(people), 5),
        "has_rels": bool(rels),
        # an entry whose buried person is never named ("un parvulo", "un niño")
        # is a distinct and common hard case
        "unnamed_principal": any(not (p.get("name") or "").strip()
                                 for p in people if str(p.get("id")) in principals),
        "enslaved": any(p.get("free") is False for p in people),
        "multi_event": len({str(ev.get("type")) for ev in events}) > 1,
        "short": len(text) < 300,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--outdir", default="production/luna_v3/burial_gold")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json")))
    rows = list(burial_entries(paths))
    with_burials = sorted({vol for vol, *_ in rows})
    print(f"{len(rows)} entries containing a burial event")
    print(f"  burials appear in {len(with_burials)} of {len(paths)} volumes: "
          f"{', '.join(with_burials)}")
    per_vol = Counter(vol for vol, *_ in rows)
    print(f"  per volume: {dict(per_vol.most_common())}")

    # bucket by profile, then take one per bucket in descending bucket rarity so
    # the rare shapes are represented before the common ones fill the quota
    buckets = {}
    for vol, e, data, events in rows:
        key = tuple(sorted(profile(e, data, events).items()))
        buckets.setdefault(key, []).append((vol, e, data, events))
    print(f"{len(buckets)} distinct entry profiles")

    import random
    rng = random.Random(args.seed)
    vols = sorted({vol for vol, *_ in rows})

    # Volume is the OUTER axis, not a filter applied afterwards.
    #
    # Two earlier versions of this treated it as a cap over a globally ordered
    # bucket list, and both collapsed onto one register: 201991 holds 1,837 of
    # the 2,024 burials, so it owns nearly every bucket whether the list is
    # sorted by rarity or by size, and the cap could only reject candidates that
    # were never offered. Allocating per volume first makes the guarantee
    # structural.
    #
    # The split is deliberately near-equal rather than proportional. Proportional
    # would be 11 and 1, which tells us nothing about the second register --
    # different scribe, different formulae, and in 701054's case a different
    # language. For a gold set, covering both hands matters more than mirroring
    # the corpus mix.
    quota = {v: args.n // len(vols) for v in vols}
    for v in vols[:args.n % len(vols)]:
        quota[v] += 1

    picked = []
    for vol in vols:
        vol_buckets = {}
        for key, members in buckets.items():
            mine = [m for m in members if m[0] == vol]
            if mine:
                vol_buckets[key] = mine
        by_size = sorted(vol_buckets, key=lambda k: (-len(vol_buckets[k]), str(k)))
        want = min(quota[vol], len(by_size))
        # Half from this volume's COMMONEST profiles, half from its rarest.
        #
        # Rarity-first alone is a trap, and it caught this script: the rarest
        # burial profile is "no date at all", so one version filled six of twelve
        # slots from the 16 undated entries in a 1,837-entry register. Gold built
        # only from outliers measures the extractor on outliers; gold built only
        # from the modal formula misses everything that breaks it.
        half = want // 2
        order, seen = [], set()
        for k in by_size[:half] + list(reversed(by_size))[: want - half]:
            if k not in seen:
                seen.add(k)
                order.append(k)
        for key in order[:want]:
            cands = sorted(vol_buckets[key], key=lambda c: c[1]["id"])
            picked.append((key, cands[0] if len(cands) == 1
                           else rng.choice(cands[:3])))
        if want < quota[vol]:
            print(f"  note: {vol} has only {len(by_size)} distinct profiles, "
                  f"so it contributes {want} of its {quota[vol]} slots")

    os.makedirs(args.outdir, exist_ok=True)
    examples = []
    for key, (vol, e, data, events) in picked:
        text = e.get("text_faithful") or ""
        examples.append({
            "type": "burial",
            "language": language_of(text),
            "country": "REVIEW",          # per-volume metadata Daniel holds, not us
            "state": "REVIEW",
            "city": "REVIEW",
            "institution": "REVIEW",
            "id": int(vol) if vol.isdigit() else vol,
            "entry": e["id"],
            "raw": text,
            "normalized": e.get("normalized") or "",
            "data": data,
            "_provenance": "MODEL OUTPUT, NOT GOLD -- correct in place, then delete "
                           "this key and the _profile key",
            "_profile": dict(key),
        })

    out = os.path.join(args.outdir, "burial_gold_candidates.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"examples": examples}, f, ensure_ascii=False, indent=1)

    print(f"\nselected {len(examples)}:")
    for x in examples:
        p = x["_profile"]
        flags = ",".join(k for k in ("unnamed_principal", "enslaved", "multi_event",
                                     "short", "has_rels") if p.get(k))
        print(f"  {x['entry']:>18}  {p['lang']:<10} {p['decade'] or '?':>5}  "
              f"{p['n_people']} people  {flags}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
