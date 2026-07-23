#!/usr/bin/env python3
"""assemble_corpus.py — post-batch, OFFLINE ($0) assembly of the Luna corpus.

Run this AFTER the monitor has downloaded and validated the provider results
(the `production/luna_live/*.output.jsonl` files). It performs no network calls
and reads no API key. For every sacramental volume it:

  1. groups the downloaded provider response rows by volume (via custom_id),
     splitting the single big multi-volume Batch job as well as per-batch files;
  2. materializes records — faithful text/images/partial from the deterministic
     corpus, `normalized`+`data` from the validated Luna rows — refusing to
     silently drop: invalid or missing records are counted and reported;
  3. runs the free QA -> identity -> graph pipeline per volume;
  4. runs one cross-volume pipeline (people linked across volumes);
  5. writes production/luna_live/CORPUS_SUMMARY.json.

    python assemble_corpus.py [--live production/luna_live] [--corpus production/corpus]

Coverage < 100% for a volume is reported, never hidden. Nothing here spends
money or can submit paid work.
"""
import argparse
import glob
import json
import re
from pathlib import Path

from ssda_nlp_tools.batch_extract import parse_response

VOLUMES = ["176899", "201991", "29597", "375062", "701054"]
_VOL_RE = re.compile(r"(" + "|".join(VOLUMES) + r")")


def _volume_of(custom_id: str):
    m = _VOL_RE.search(custom_id or "")
    return m.group(1) if m else None


def read_rows_by_volume(live: Path):
    """{volume: {"valid": {id: {normalized,data}}, "invalid":[custom_id], "seen":set}}"""
    by = {v: {"valid": {}, "invalid": [], "batches": 0} for v in VOLUMES}
    for path in sorted(live.glob("*.output.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            vol = _volume_of(row.get("custom_id", ""))
            if vol is None:
                continue
            by[vol]["batches"] += 1
            resp = row.get("response") or {}
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if resp.get("status_code") != 200 or len(choices) != 1 \
                    or choices[0].get("finish_reason") != "stop":
                by[vol]["invalid"].append(row.get("custom_id"))
                continue
            text = choices[0].get("message", {}).get("content")
            try:
                values, missing = parse_response(text, [], validate=True)
            except Exception:
                by[vol]["invalid"].append(row.get("custom_id"))
                continue
            if missing:
                by[vol]["invalid"].append(row.get("custom_id"))
                continue
            by[vol]["valid"].update(values)
    return by


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=Path, default=Path("production/luna_live"))
    ap.add_argument("--corpus", type=Path, default=Path("production/corpus"))
    args = ap.parse_args(argv)

    import materialize_luna_results as M
    import run_pipeline

    by = read_rows_by_volume(args.live)
    outdir = args.live / "assembled"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"volumes": {}, "totals": {}}
    materialized_files = []
    tot_corpus = tot_mat = tot_missing = tot_invalid = 0

    for vol in VOLUMES:
        corpus_path = args.corpus / f"{vol}.segmented.json"
        if not corpus_path.exists():
            continue
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        extracted = by[vol]["valid"]
        if not extracted:
            summary["volumes"][vol] = {"state": "no provider output yet",
                                       "corpus_records": len(corpus.get("entries", []))}
            continue
        result = M.materialize(corpus, extracted, allow_incomplete=True)
        mat_path = outdir / f"{vol}.materialized.json"
        mat_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        materialized_files.append((vol, mat_path))
        cov = result["coverage"]
        # per-volume QA/identity/graph
        pipe_dir = args.live / f"{vol}_final_pipeline"
        run_pipeline.main([str(mat_path), "--tag", vol, "--outdir", str(pipe_dir)])
        complete = cov["missing_records"] == 0 and not by[vol]["invalid"]
        summary["volumes"][vol] = {
            "state": "COMPLETE" if complete else "PARTIAL",
            "corpus_records": cov["corpus_records"],
            "materialized_records": cov["materialized_records"],
            "missing_records": cov["missing_records"],
            "invalid_batches": len(by[vol]["invalid"]),
            "pipeline": str(pipe_dir)}
        tot_corpus += cov["corpus_records"]; tot_mat += cov["materialized_records"]
        tot_missing += cov["missing_records"]; tot_invalid += len(by[vol]["invalid"])
        print(f"{vol}: {cov['materialized_records']}/{cov['corpus_records']} records "
              f"({'COMPLETE' if complete else 'PARTIAL'}; missing {cov['missing_records']}, "
              f"invalid batches {len(by[vol]['invalid'])})")

    # cross-volume linkage (people linked ACROSS volumes) once >1 volume present
    if len(materialized_files) > 1:
        corpus_dir = args.live / "corpus_final_pipeline"
        run_pipeline.main([str(p) for _, p in materialized_files]
                          + ["--tag", "CORPUS", "--outdir", str(corpus_dir)])
        summary["corpus_pipeline"] = str(corpus_dir)

    summary["totals"] = {"corpus_records": tot_corpus, "materialized_records": tot_mat,
                         "missing_records": tot_missing, "invalid_batches": tot_invalid,
                         "volumes_with_output": len(materialized_files)}
    (args.live / "CORPUS_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTOTAL materialized: {tot_mat}/{tot_corpus} records; "
          f"missing {tot_missing}; invalid batches {tot_invalid}")
    print(f"-> {args.live / 'CORPUS_SUMMARY.json'}")
    if tot_missing or tot_invalid:
        print("NOTE: coverage < 100% — see CORPUS_SUMMARY.json; missing/invalid are "
              "reported for repair, never silently dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
