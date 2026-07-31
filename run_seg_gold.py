#!/usr/bin/env python3
"""run_seg_gold.py — our segmenter against human entry boundaries.

Offline, $0, no network, no key.

    python run_seg_gold.py

SSDA's hand transcriptions are stored one record per ENTRY (`0033-01`, `0033-02`),
so they are entry-boundary ground truth as well as text ground truth. That is
the only human-labelled segmentation we have; every previous segmentation number
in this project was measured against an LLM-generated reference, which cannot
tell us whether both agree and are wrong together.

WHY THIS RUNS ON MACHINE TEXT AND NOT ON THE HUMAN TEXT
-------------------------------------------------------
The obvious cheaper experiment is to concatenate the human entries back into a
page and ask the segmenter to recover the boundaries. That would isolate the
segmenter from transcription error and would work across all nine hand
transcribed volumes rather than the three that overlap.

It is worthless, and measurably so. Checked before building it: human entries
contain ZERO newlines in all 3,452 of them, while machine transcription carries
about 35 per page. Concatenating with a newline therefore hands the segmenter
exactly one line per entry, and the segmenter's first move is to classify lines.
It would score near-perfectly on a task nobody asked it to do.

So the reference is the human entry list and the input is the machine page text,
which means this measures transcription and segmentation TOGETHER. That is the
honest version and it is also the one that matches production.

WHY THIS COUNTS ENTRIES PER PAGE INSTEAD OF MATCHING ENTRY TEXT
---------------------------------------------------------------
The first version matched each human entry to the segmenter's most similar
predicted entry, which is how segeval works everywhere else in this repo. It
reported recall 0.127 and it was measuring nothing.

The reason is that these registers are formulaic. Two DIFFERENT baptisms from
the same parish, same priest and same year share almost all of their text: the
church, the officiant, the sacramental boilerplate, the closing formula. Checked
on 419324, the median human entry's best match among the predictions scored
0.607 -- and reading that pair shows one is Ana Joaquina born in February and
the other Anselma Josefa born in December. Different people, different entries,
0.607 similar.

Meanwhile the SAME entry, transcribed independently by a person and by Gemini,
differs by roughly 6% of its characters plus abbreviation and accent
conventions. So the same-entry and different-entry similarity distributions
overlap almost completely, and no threshold separates them: at 0.55 the eval
matches 45 of 66 entries, mostly to the wrong ones, and at 0.75 it matches 11.
Text similarity is simply not identifiable across two transcriptions of a
formulaic register, and lowering the threshold buys false matches, not recall.

So the metric here is ENTRIES PER PAGE, which needs no text agreement at all.
The human file says folio 0033 holds four entries; the segmenter either finds
four or it does not. That is the same logic as the margin-number check in
segeval, which works for exactly the same reason.

Because entries run across folios, "entries on page P" means entries STARTING on
page P for both sides.
"""
import argparse
import json
import os
import re
from collections import Counter

from ssda_nlp_tools.manual_gold import (human_page_of, human_pages, machine_pages,
                                        offset_map)
from ssda_nlp_tools.segment import segment_volume
from ssda_nlp_tools.transcription_integrity import check_page

# A transcriber's note that the record is unreadable, not a record.
_ILLEGIBLE = re.compile(r"(totalmente\s+desva[ií]d[oa]|ilegible|illegible|roto|en\s+blanco)\.?", re.I)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", default="../ssda-openai/json")
    ap.add_argument("--machine", default="../transcriptions/json")
    ap.add_argument("--out", default="production/luna_v3/seg_gold.json")
    args = ap.parse_args(argv)

    report = {}
    for name in sorted(os.listdir(args.manual)):
        vol = name[:-5]
        mpath = os.path.join(args.machine, name)
        if not name.endswith(".json") or not os.path.exists(mpath):
            continue
        h = json.load(open(os.path.join(args.manual, name), encoding="utf-8"))
        m = json.load(open(mpath, encoding="utf-8"))
        hp, mp = human_pages(h), machine_pages(m)
        offs = offset_map(hp, mp)

        # Only the pages a human actually transcribed, mapped through the drift
        # offset. Feeding the whole volume would count every untranscribed page's
        # entries as false positives.
        pages, seen = [], set()
        for pid in sorted(hp):
            tgt = str(int(pid) + offs.get(pid, 0)).zfill(4)
            if tgt in mp and tgt not in seen and check_page(mp[tgt])["ok"]:
                seen.add(tgt)
                pages.append((f"{vol}-{tgt}.jpg", mp[tgt]))
        if not pages:
            continue

        # Human entry starts per folio, minus the illegibility placeholders.
        # The transcribers record an unreadable record as an entry whose whole
        # text is "totalmente desvaido". That is a real entry historically and a
        # convention difference for us: our segmenter cannot emit an entry from
        # text that was never transcribed. 18 of 975 in 15834, so excluding them
        # changes little -- which is the point of counting them rather than
        # assuming. Matched by pattern, not by a length cutoff, because a length
        # cutoff would also silently drop genuinely short records.
        human_starts, skipped = Counter(), 0
        for e in h.get("entries") or []:
            pid = human_page_of(e.get("id"))
            if pid not in hp:
                continue
            if _ILLEGIBLE.fullmatch((e.get("raw") or "").strip()):
                skipped += 1
                continue
            human_starts[pid] += 1

        seg = segment_volume(pages)
        # Segmenter entry starts per folio. A leading fragment continuing from
        # the previous page is not a start, which is why per_image entries are
        # used rather than the merged volume list.
        pred_starts = {pg["image"]: len(pg.get("entries") or [])
                       for pg in seg.get("per_image") or []}

        rows, exact, off_by_one, hsum, psum = [], 0, 0, 0, 0
        for pid in sorted(hp):
            tgt = str(int(pid) + offs.get(pid, 0)).zfill(4)
            img = f"{vol}-{tgt}.jpg"
            if img not in pred_starts:
                continue
            hn, pn = human_starts.get(pid, 0), pred_starts[img]
            hsum += hn
            psum += pn
            exact += (hn == pn)
            off_by_one += (abs(hn - pn) == 1)
            rows.append({"page": pid, "machine_page": tgt,
                         "human": hn, "predicted": pn, "delta": pn - hn})

        n = len(rows) or 1
        rep = {"pages": len(rows), "human_entries": hsum, "predicted_entries": psum,
               "illegible_excluded": skipped,
               "exact": exact, "exact_rate": round(exact / n, 4),
               "within_one": exact + off_by_one,
               "within_one_rate": round((exact + off_by_one) / n, 4),
               "over_split": sum(1 for r in rows if r["delta"] > 0),
               "under_split": sum(1 for r in rows if r["delta"] < 0),
               "mean_abs_error": round(sum(abs(r["delta"]) for r in rows) / n, 3),
               "rows": rows}
        report[vol] = rep
        print(f"--- {vol}: {rep['pages']} folios, {hsum} human entries, "
              f"{psum} found   ({skipped} illegible placeholders excluded)")
        print(f"    exact entry count      {exact:4d}/{rep['pages']}  "
              f"({100*rep['exact_rate']:.1f}%)")
        print(f"    within one entry       {rep['within_one']:4d}/{rep['pages']}  "
              f"({100*rep['within_one_rate']:.1f}%)")
        print(f"    over-split {rep['over_split']:4d}   under-split "
              f"{rep['under_split']:4d}   mean abs error {rep['mean_abs_error']}")
        print()

    tp = sum(r["pages"] for r in report.values())
    te = sum(r["exact"] for r in report.values())
    tw = sum(r["within_one"] for r in report.values())
    th = sum(r["human_entries"] for r in report.values())
    tq = sum(r["predicted_entries"] for r in report.values())
    print(f"=== OVERALL over {len(report)} volumes, {tp} folios ===")
    print(f"    human entries {th:,}   segmenter found {tq:,}   "
          f"({100*tq/max(th,1):.1f}% of ground truth)")
    print(f"    exact entry count per folio  {te}/{tp}  ({100*te/max(tp,1):.1f}%)")
    print(f"    within one                   {tw}/{tp}  ({100*tw/max(tp,1):.1f}%)")
    print("\n    Transcription and segmentation together, on the 3 volumes that")
    print("    have both. Counts, not text matching: in a formulaic register two")
    print("    different entries are ~0.61 similar, so text matching across two")
    print("    independent transcriptions is not identifiable at any threshold.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
