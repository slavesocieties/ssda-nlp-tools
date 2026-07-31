#!/usr/bin/env python3
"""run_manual_gold.py — how accurate is our transcription, really?

Offline, $0, no network, no key.

    python run_manual_gold.py --manual ../ssda-openai/json --machine ../transcriptions/json

Compares Archivault/Gemini page transcriptions against SSDA's own hand
transcriptions for every volume that exists in both. See ssda_nlp_tools.manual_gold
for what the comparison is and is not fair about; the short version is that
substitution rate is the quality signal, deletion rate is the alarm, and
insertion rate is mostly page-scope difference because the human transcribed
entries while the machine transcribed the whole folio.
"""
import argparse
import glob
import json
import os

from ssda_nlp_tools.manual_gold import (aggregate, align_pages, entry_counts,
                                        human_pages, machine_pages)
from ssda_nlp_tools.transcription_integrity import check_page


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", default="../ssda-openai/json")
    ap.add_argument("--machine", default="../transcriptions/json")
    ap.add_argument("--out", default="production/luna_v3/manual_gold.json")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args(argv)

    manual = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(args.manual, "*.json"))}
    machine = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(args.machine, "*.json"))}
    shared = sorted(set(manual) & set(machine))
    print(f"hand transcribed volumes : {len(manual)}")
    print(f"machine transcribed      : {len(machine)}")
    print(f"BOTH (measurable)        : {len(shared)} -> {shared}\n")
    if not shared:
        ap.error("no overlap: nothing can be measured")

    report, all_rows = {}, []
    for vol in shared:
        h = json.load(open(manual[vol], encoding="utf-8"))
        m = json.load(open(machine[vol], encoding="utf-8"))
        hp, mp = human_pages(h), machine_pages(m)
        # A page the transcriber never produced is not a transcription error;
        # averaging its 100% deletion into an accuracy figure describes an
        # outage, not quality. Counted separately, excluded from the rates.
        hard = {p for p, t in mp.items() if not check_page(t)["ok"]}
        res = align_pages(hp, mp)
        res["hard_failures"] = sorted(hard & set(hp))
        res["pages"] = [r for r in res["pages"] if r["machine_page"] not in hard]
        agg = aggregate(res["pages"])
        res["aggregate"] = agg
        res["human_entries"] = sum(entry_counts(h).values())
        report[vol] = res
        all_rows.extend(res["pages"])

        print(f"--- {vol}  ({res['human_entries']} hand-transcribed entries)")
        print(f"    pages compared {agg['pages']:4d}   "
              f"human chars {agg['human_chars']:,}")
        print(f"    realigned for drift {res['pages_realigned']:4d}   "
              f"offsets {res['offsets_used']}")
        print(f"    excluded, transcriber failed outright: "
              f"{len(res['hard_failures'])}")
        print(f"    substitution {100*agg['sub_rate']:6.2f}%   <- quality signal")
        print(f"    deletion     {100*agg['del_rate']:6.2f}%   <- machine missed "
              f"text a human read")
        print(f"    insertion    {100*agg['ins_rate']:6.2f}%   <- mostly page scope, "
              f"not error")
        print(f"    median page similarity {agg['median_similarity']:.3f}  "
              f"median CER ignoring spaces {agg['median_cer_nospace']:.3f}")
        if res["suspect_alignment"]:
            print(f"    !! {len(res['suspect_alignment'])} pages align so poorly the "
                  f"PAIRING is suspect: {res['suspect_alignment'][:8]}")
        if res["human_only_pages"]:
            print(f"    {len(res['human_only_pages'])} hand-transcribed pages have no "
                  f"machine transcription")

    overall = aggregate(all_rows)
    print(f"\n=== OVERALL over {overall['pages']} pages, "
          f"{overall['human_chars']:,} human characters ===")
    print(f"    substitution {100*overall['sub_rate']:6.2f}%")
    print(f"    deletion     {100*overall['del_rate']:6.2f}%")
    print(f"    insertion    {100*overall['ins_rate']:6.2f}%")
    print(f"    median page similarity {overall['median_similarity']:.3f}")

    worst = sorted(all_rows, key=lambda r: r["similarity"])[:args.worst]
    print(f"\n--- {len(worst)} worst pages by similarity (read these before "
          f"believing any average)")
    for r in worst:
        print(f"    page {r['page']}  sim {r['similarity']:.3f}  "
              f"sub {100*r['sub_rate']:5.1f}%  del {100*r['del_rate']:5.1f}%  "
              f"ins {100*r['ins_rate']:6.1f}%  human {r['human_chars']:,}c")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"volumes": report, "overall": overall}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
