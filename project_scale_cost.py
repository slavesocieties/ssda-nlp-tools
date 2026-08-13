"""What does finishing cost? Projected from measured numbers, with the unknowns named.

HANDOVER §2 says 7 of 3,900 volumes, 0.18%. That is a true statement nobody can
act on. This turns it into money and wall-clock, so the remaining 99.82% becomes
a decision someone can fund rather than a caveat at the top of a document.

EVERY INPUT IS LABELLED. Measured numbers come from this machine; projections say
what they assume; and the one genuinely unknown cost is named as unknown rather
than filled with a plausible number. HANDOVER §9 rule 7 -- name the unverified --
matters more here than anywhere else, because a cost estimate is the kind of
figure that gets quoted in a budget request and never re-checked.

WHAT THIS DELIBERATELY DOES NOT DO
    It does not estimate stage 2 (transcription). That runs in Archivault, is
    external to this repo, and no per-page price has ever been measured here.
    Inventing one would put a fabricated number in the largest line item.
"""
from __future__ import annotations

import argparse
import glob
import json

# ---- MEASURED ON THIS MACHINE ---------------------------------------------
VOLUMES_TOTAL = 3_900          # ssda-openai/volumes.json, verified 2026-08-12
VOLUMES_DONE = 7
# production spend for the delivered corpus, from the v3 build
CORPUS_SPEND_USD = 24.53
CORPUS_RECORDS = 5_228
# cheapest model in the live bake-off, $/entry, 32/32 correct
BAKEOFF_BEST_USD_PER_ENTRY = 0.0036
# merge, keyed blocking, deduped corpus, clean serial run
MERGE_SECONDS = 796
MERGE_PAIRS = 14_472_013
# projected full-collection pair count after keyed blocking (HANDOVER §7a)
FULL_PAIRS = 1.33e10
CORES = 24


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    a = ap.parse_args(argv)

    entries = 0
    vols = 0
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries += len(json.load(open(p, encoding="utf-8"))["entries"])
        vols += 1
    if not vols:
        raise SystemExit(f"no volumes under {a.assembled!r}")

    per_vol = entries / vols
    remaining_vols = VOLUMES_TOTAL - VOLUMES_DONE
    projected_entries = per_vol * VOLUMES_TOTAL
    remaining_entries = per_vol * remaining_vols

    print("MEASURED")
    print(f"  volumes processed          : {vols}")
    print(f"  entries in them            : {entries:,}")
    print(f"  entries per volume         : {per_vol:,.0f}")
    print(f"  production spend so far    : ${CORPUS_SPEND_USD:,.2f} "
          f"over {CORPUS_RECORDS:,} records")
    usd_per_entry = CORPUS_SPEND_USD / CORPUS_RECORDS
    print(f"  -> extraction cost/entry   : ${usd_per_entry:.5f} "
          f"(bake-off best was ${BAKEOFF_BEST_USD_PER_ENTRY:.4f})")
    print(f"  merge, 7 volumes           : {MERGE_SECONDS}s for "
          f"{MERGE_PAIRS:,} pairs")

    print("\nPROJECTED TO THE FULL COLLECTION")
    print(f"  volumes remaining          : {remaining_vols:,} "
          f"({100*remaining_vols/VOLUMES_TOTAL:.2f}% of the archive)")
    print(f"  entries remaining          : ~{remaining_entries:,.0f}")
    print(f"  entries in total           : ~{projected_entries:,.0f}")

    print("\n  STAGE 4, extraction -- the cost this repo controls")
    for label, rate in (("at the measured production rate", usd_per_entry),
                        ("at the cheapest bake-off rate ", BAKEOFF_BEST_USD_PER_ENTRY)):
        print(f"    {label} : ${remaining_entries * rate:,.0f}")
    print("    (Batch API halves this on both rows; the delivered corpus was")
    print("     built that way, so treat the figures as the un-batched ceiling.)")

    print("\n  STAGE 5, disambiguation -- compute, not money")
    hours = FULL_PAIRS / (MERGE_PAIRS / MERGE_SECONDS) / 3600
    print(f"    projected pairs            : {FULL_PAIRS:,.0f} (keyed blocking)")
    print(f"    single core                : ~{hours:,.0f} hours")
    print(f"    {CORES} cores                   : ~{hours/CORES:,.0f} hours")
    print("    Without keyed blocking this is ~7.46e12 pairs and does not")
    print("    finish. That switch is the difference between a weekend and a")
    print("    year, and it is already the default.")

    print("\nUNKNOWN, AND THE LARGEST LINE ITEM")
    print("  STAGE 2, transcription. Runs in Archivault, outside this repo, and")
    print("  no per-page price has ever been measured here. At ~268 pages per")
    print("  volume that is roughly 1.05M pages for the collection. Whatever")
    print("  that costs per page dominates everything above -- extraction is")
    print("  cents per volume and transcription is a whole-page vision call per")
    print("  page. GET THIS NUMBER BEFORE QUOTING A TOTAL.")

    print("\nWHAT THIS MEANS")
    print("  The engineering is not the bottleneck and has not been for a while.")
    print("  Stage 5 went from not-finishing to ~half a day when keyed blocking")
    print("  landed. Stage 4 is a four-figure sum at worst. The project is")
    print("  0.18% done because nobody has bought 1.05M pages of transcription,")
    print("  not because the pipeline cannot take them.")
    print("\n  The other bottleneck is a person: recall, end-to-end accuracy,")
    print("  burial gold and two vocabulary rulings all wait on Daniel, and no")
    print("  amount of compute substitutes for any of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
