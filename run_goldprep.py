#!/usr/bin/env python3
"""run_goldprep.py — bootstrap a gold-labeling sheet from a model run.

    python run_goldprep.py VOLUME.json --out gold_draft.json
        [--type baptism --country "United States" --state Florida
         --city "San Agustin" --institution parish] [--limit N]

Writes a training_data-format file where every entry is PRE-FILLED with the
model's own extraction. A human then only has to CORRECT the data blocks (much
cheaper than authoring them), delete any entry too damaged to trust, and the
result can be appended to training_data.json — growing both the few-shot pool
and the eval gold set at once. The sheet carries a per-entry "_review" field
("pending" -> set to "done" when checked) so partial progress is usable:
`run_eval.py` ignores unreviewed entries only if you filter them yourself, so
keep draft and final files separate. No API keys, no network.
"""
import argparse
import json

from ssda_nlp_tools.evaluate import load_entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", dest="rtype", default="baptism")
    ap.add_argument("--language", default="Spanish")
    ap.add_argument("--country", default="United States")
    ap.add_argument("--state", default="Florida")
    ap.add_argument("--city", default="San Agustin")
    ap.add_argument("--institution", default="parish")
    ap.add_argument("--id", dest="vol_id", default=None, help="volume id for the sheet")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    with open(args.volume, "r", encoding="utf-8") as f:
        src = json.load(f)
    rows = src.get("examples") or src.get("entries") or []
    if args.limit:
        rows = rows[: args.limit]

    examples = []
    for r in rows:
        examples.append({
            "_review": "pending",       # set to "done" after human correction
            "type": r.get("type", args.rtype),
            "language": r.get("language", args.language),
            "country": r.get("country", args.country),
            "state": r.get("state", args.state),
            "city": r.get("city", args.city),
            "institution": r.get("institution", args.institution),
            "id": r.get("id", args.vol_id) or args.vol_id or "",
            "entry": r.get("entry") or r.get("id") or "",
            "raw": r.get("raw", ""),
            "normalized": r.get("normalized", ""),
            "data": r.get("data", {"people": [], "events": []}),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"examples": examples}, f, ensure_ascii=False, indent=2)
    n_people = sum(len(e["data"].get("people", [])) for e in examples)
    print(f"{len(examples)} entries pre-filled ({n_people} people) -> {args.out}")
    print("correct each entry's data block, set \"_review\": \"done\", then append the")
    print("finished examples to training_data.json (drop the _review key).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
