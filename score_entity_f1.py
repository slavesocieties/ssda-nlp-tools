#!/usr/bin/env python3
"""score_entity_f1.py — offline entity-level F1 of saved bake-off outputs vs reference,
aggregated across every volume we have. No network, no keys, no spend.

For each (volume-reference, bake-off-output) pair it reshapes the model's saved
`results` into the {"examples":[...]} form run_eval expects, reports explicit
entry coverage, and scores missing/spurious entries as false negatives/positives.
Corpus scores pool TP/FP/FN across volumes before calculating a true
micro-average.

    python score_entity_f1.py

Add pairs to VOLUMES as more bake-off outputs are produced. A pair is skipped
(with a note) if its output file is absent, so this is safe to run any time.
"""
import glob
import json
from pathlib import Path

from ssda_nlp_tools.evaluate import evaluate
from ssda_nlp_tools.textmatch import sum_prf

ROOT = Path(__file__).resolve().parent

# (volume tag, gold file, bake-off output patterns; all matches are merged)
VOLUMES = [
    ("0035_0044", "Sample_output/Generated_0035_0044_4o_prompt_V2.json",
     ["openai_bakeoff_results.json", "gpt56luna_b*.json", "claudehaiku_b*.json",
      "entity_0035_*_b*.json", "*0035*bakeoff_results.json"]),
    ("0013_0023", "Sample_output/Generated_0013_0023_4o_prompt_V2.json",
     ["entity_0013_*_b*.json", "*0013*bakeoff_results.json"]),
    ("0024_0034", "Sample_output/Generated_0024_0034_4o_prompt_V2.json",
     ["entity_0024_*_b*.json", "*0024*bakeoff_results.json"]),
]
DIMS = ["people", "events", "relationships"]


def _find_all(patterns):
    found = []
    for pat in patterns:
        hits = sorted(glob.glob(str(ROOT / pat)))
        for hit in hits:
            if hit not in found:
                found.append(hit)
    return found


def _model_predictions(bakeoff_paths):
    """{model: {entry_id: {entry, normalized, data}}} from a bake-off output."""
    out = {}
    for bakeoff_path in bakeoff_paths:
        data = json.loads(Path(bakeoff_path).read_text(encoding="utf-8"))
        for model, row in data.get("models", {}).items():
            if row.get("status") == "skipped":
                continue
            preds = out.setdefault(model, {})
            for b in row.get("batches", []):
                for r in b.get("results", []):
                    eid = str(r.get("entry"))
                    if eid and "data" in r and eid not in preds:
                        preds[eid] = {"entry": eid, "normalized": r.get("normalized", ""),
                                      "data": r.get("data") or {}}
    return out


def main():
    print(f"{'volume':10s} {'model':18s} " + " ".join(f"{d[:5]:>7s}" for d in DIMS)
          + "   coverage")
    print("-" * 82)
    per_model = {}
    for tag, gold_rel, patterns in VOLUMES:
        gold_path = ROOT / gold_rel
        bakeoffs = _find_all(patterns)
        if not bakeoffs:
            print(f"{tag:10s} (no bake-off output found yet — run the model on this volume)")
            continue
        gold = {str(e["entry"]): e for e in
                json.loads(gold_path.read_text(encoding="utf-8"))["examples"]}
        for model, preds in sorted(_model_predictions(bakeoffs).items()):
            covered = set(preds) & set(gold)
            g = {"examples": list(gold.values())}
            p = {"examples": list(preds.values())}
            rep = evaluate(g, p)
            f1 = {d: (rep.get(d) or {}).get("f1") for d in DIMS}
            cells = " ".join((f"{f1[d]:7.3f}" if f1[d] is not None else f"{'—':>7s}") for d in DIMS)
            print(f"{tag:10s} {model:18s} {cells}   {len(covered)}/{len(gold)}")
            row = per_model.setdefault(model, {
                "dimensions": {d: [] for d in DIMS},
                "returned": 0,
                "reference": 0,
                "extra": 0,
            })
            for d in DIMS:
                if f1[d] is not None:
                    row["dimensions"][d].append(rep[d])
            row["returned"] += len(covered)
            row["reference"] += len(gold)
            row["extra"] += len(set(preds) - set(gold))
    print("-" * 82)
    print("corpus micro-average (pooled TP/FP/FN; missing entries included):")
    for model, row in sorted(per_model.items()):
        cells = []
        for d in DIMS:
            xs = row["dimensions"][d]
            if xs:
                cells.append(f"{sum_prf(xs)['f1']:7.3f}")
            else:
                cells.append(f"{'—':>7s}")
        coverage = f"{row['returned']}/{row['reference']}"
        if row["extra"]:
            coverage += f" (+{row['extra']} extra)"
        print(f"{'ALL':10s} {model:18s} {' '.join(cells)}   {coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
