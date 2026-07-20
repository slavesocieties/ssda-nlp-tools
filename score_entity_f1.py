#!/usr/bin/env python3
"""score_entity_f1.py — offline entity-level F1 of saved bake-off outputs vs gold,
aggregated across every volume we have. No network, no keys, no spend.

For each (volume-gold, bake-off-output) pair it reshapes the model's saved
`results` into the {"examples":[...]} form run_eval expects, aligns on the
entries the model actually produced (so we score extraction quality, not
coverage), and reports people/events/relationships F1 per model per volume plus
a corpus-wide micro-average.

    python score_entity_f1.py

Add pairs to VOLUMES as more bake-off outputs are produced. A pair is skipped
(with a note) if its output file is absent, so this is safe to run any time.
"""
import glob
import json
import os
from pathlib import Path

from ssda_nlp_tools.evaluate import evaluate

ROOT = Path(__file__).resolve().parent

# (volume tag, gold file, [candidate bake-off output files, first found wins])
VOLUMES = [
    ("0035_0044", "Sample_output/Generated_0035_0044_4o_prompt_V2.json",
     ["openai_bakeoff_results.json", "*0035*bakeoff_results.json"]),
    ("0013_0023", "Sample_output/Generated_0013_0023_4o_prompt_V2.json",
     ["*0013*bakeoff_results.json"]),
    ("0024_0034", "Sample_output/Generated_0024_0034_4o_prompt_V2.json",
     ["*0024*bakeoff_results.json"]),
]
DIMS = ["people", "events", "relationships"]


def _find(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(str(ROOT / pat)))
        if hits:
            return hits[0]
    return None


def _model_predictions(bakeoff_path):
    """{model: {entry_id: {entry, normalized, data}}} from a bake-off output."""
    data = json.loads(Path(bakeoff_path).read_text(encoding="utf-8"))
    out = {}
    for model, row in data.get("models", {}).items():
        if row.get("status") == "skipped":
            continue
        preds = {}
        for b in row.get("batches", []):
            for r in b.get("results", []):
                eid = str(r.get("entry"))
                if eid and "data" in r and eid not in preds:
                    preds[eid] = {"entry": eid, "normalized": r.get("normalized", ""),
                                  "data": r.get("data") or {}}
        if preds:
            out[model] = preds
    return out


def main():
    gold_all = {d: [] for d in DIMS}   # for a simple macro-average across volumes
    print(f"{'volume':10s} {'model':18s} " + " ".join(f"{d[:5]:>7s}" for d in DIMS) + "   n")
    print("-" * 70)
    per_model = {}
    for tag, gold_rel, patterns in VOLUMES:
        gold_path = ROOT / gold_rel
        bo = _find(patterns)
        if not bo:
            print(f"{tag:10s} (no bake-off output found yet — run the model on this volume)")
            continue
        gold = {str(e["entry"]): e for e in
                json.loads(gold_path.read_text(encoding="utf-8"))["examples"]}
        for model, preds in sorted(_model_predictions(bo).items()):
            covered = [e for e in preds if e in gold]
            g = {"examples": [gold[e] for e in covered]}
            p = {"examples": [preds[e] for e in covered]}
            rep = evaluate(g, p)
            f1 = {d: (rep.get(d) or {}).get("f1") for d in DIMS}
            cells = " ".join((f"{f1[d]:7.3f}" if f1[d] is not None else f"{'—':>7s}") for d in DIMS)
            print(f"{tag:10s} {model:18s} {cells}   {len(covered)}")
            row = per_model.setdefault(model, {d: [] for d in DIMS})
            for d in DIMS:
                if f1[d] is not None:
                    row[d].append((f1[d], len(covered)))
    print("-" * 70)
    print("weighted mean across volumes (by entries scored):")
    for model, row in sorted(per_model.items()):
        cells = []
        for d in DIMS:
            xs = row[d]
            if xs:
                w = sum(n for _, n in xs)
                cells.append(f"{sum(v * n for v, n in xs) / w:7.3f}")
            else:
                cells.append(f"{'—':>7s}")
        print(f"{'ALL':10s} {model:18s} {' '.join(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
