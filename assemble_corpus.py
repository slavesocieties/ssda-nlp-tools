#!/usr/bin/env python3
"""assemble_corpus.py — post-batch, OFFLINE ($0) assembly of the Luna corpus.

Run this AFTER the monitor has downloaded and validated the provider results
(the `production/luna_live/*.accepted.jsonl` files). It performs no network calls
and reads no API key. For every sacramental volume it:

  1. groups the downloaded provider response rows by volume (via custom_id),
     splitting the single big multi-volume Batch job as well as per-batch files;
  2. materializes records — faithful text/images/partial from the deterministic
     corpus, `normalized`+`data` from the validated Luna rows — refusing to
     silently drop: invalid or missing records are counted and reported;
  3. runs the free QA -> identity -> graph pipeline per volume;
  4. runs one cross-volume pipeline (people linked across volumes);
  5. writes production/luna_live/CORPUS_SUMMARY.json.

    python assemble_corpus.py [--live production/luna_live] \
        [--accepted-dir production/luna_v3] \
        [--accepted-dir production/new_volumes/live] \
        [--corpus production/corpus] [--corpus OTHER_SEGMENTED_DIR]

Coverage < 100% for a volume is reported, never hidden. Nothing here spends
money or can submit paid work.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

from ssda_nlp_tools.batch_extract import parse_response

# The five volumes of the original delivery. Kept only as a default for
# reporting; volume detection is NO LONGER limited to this list.
VOLUMES = ["176899", "201991", "29597", "375062", "701054"]

# A volume id is a 4-7 digit run followed by the request suffix. The lookahead
# is what makes this safe against run-id prefixes: in `v3-176899-b0000` the
# leading `3` is too short to match, and in a hypothetical `run2026-176899-b0`
# the `2026` is not followed by `-b`/`-repair`, so only the real volume wins.
#
# This replaced a hardcoded whitelist of the five delivered volumes. That
# whitelist silently returned None for 701157 and 701179 -- both already
# submitted and paid for -- and `read_rows_by_volume` skips a None volume
# without a word, so an entire extraction would have vanished at assembly with
# no error to explain it.
# `\d{4}[-.]` is the third accepted suffix and it is not cosmetic. Repair
# requests are addressed to an ENTRY, not a batch, so their custom_id is
# `v3-repair1-176899-0236-B-01`: the volume is followed by a page number rather
# than by `-b0` or `-repair`. Without it those 160 responses matched nothing,
# `read_rows_by_volume` skipped them silently, and re-assembling the corpus
# dropped it from 5,226 records to 5,066 -- all of them paid for.
#
# This is the SECOND time this mapping has silently discarded delivered work;
# the note above records the hardcoded whitelist that did the same to 701157 and
# 701179. The lesson both times is that the failure is invisible: an unmapped id
# is not an error, it is an absence. Hence the loud WARNING in
# read_rows_by_volume and the shape test in tests/test_assemble_corpus.py.
_VOL_RE = re.compile(r"(\d{4,7})-(?=b\d|repair\b|\d{4}[-.])")


def _volume_of(custom_id: str):
    """Map a provider custom_id back to its delivered volume.

    `<vol>-repair-*` intentionally maps to `<vol>`: repair requests re-fetch
    records that belong in that volume. `*-vocabtest-*` deliberately maps to
    None — those are a prompt EXPERIMENT that re-extracts entry IDs already
    present in the delivered volume, so assembling them would collide with the
    real records (same IDs) and either corrupt the volume or discard the
    experiment. They are materialized separately and compared with
    vocab_ab_report.py.
    """
    cid = custom_id or ""
    if "vocabtest" in cid:
        return None
    m = _VOL_RE.search(cid)
    return m.group(1) if m else None


def apply_delivery_convention(entries, keep_partials: bool):
    """Daniel's 2026-07-22 convention: the DELIVERED dataset drops page-truncated
    `partial` records (his references omit them). Returns (kept, dropped_count).
    The source corpus is untouched, so keep_partials=True fully reverses it."""
    if keep_partials:
        return entries, 0
    kept = [e for e in entries if not e.get("partial")]
    return kept, len(entries) - len(kept)


def discover_corpora(corpus_dirs):
    """Return ``{volume: segmented_path}`` from one or more source directories.

    Some newer segmentation artifacts predate the top-level ``volume`` field,
    so the numeric filename remains the canonical fallback.  Conflicting
    sources are rejected rather than letting directory order choose which
    faithful transcription is delivered.
    """
    found = {}
    for corpus_dir in corpus_dirs:
        for path in sorted(corpus_dir.glob("*.segmented.json")):
            filename_volume = path.name.removesuffix(".segmented.json")
            if not re.fullmatch(r"\d{4,7}", filename_volume):
                continue
            corpus = json.loads(path.read_text(encoding="utf-8"))
            declared = str(corpus.get("volume") or "")
            if declared and declared != filename_volume:
                raise ValueError(
                    f"segmented volume mismatch: {path} declares {declared}")
            previous = found.get(filename_volume)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(
                    f"duplicate segmented sources for volume {filename_volume}: "
                    f"{previous} and {path}")
            found[filename_volume] = path
    return found


def filter_provider_only(volume_rows, corpus_ids):
    """Exclude provider-returned IDs absent from the deterministic source."""
    extracted = volume_rows["valid"]
    unexpected = sorted(set(extracted) - set(corpus_ids))
    if unexpected:
        volume_rows["invalid"].extend(
            f"provider-only entry ID: {entry_id}" for entry_id in unexpected)
    return ({entry_id: value for entry_id, value in extracted.items()
             if entry_id in corpus_ids}, unexpected)


def _artifact_dirs(value):
    return [value] if isinstance(value, Path) else list(value)


def read_rows_by_volume(live):
    """{volume: {"valid": {id: {normalized,data}}, "invalid":[custom_id], "seen":set}}"""
    # Seeded with the original five so existing callers still find their keys,
    # but ANY volume present in the data is added. The previous version could
    # only ever see the five, and skipped the rest without a word.
    by = {v: {"valid": {}, "invalid": [], "rejected": [], "batches": 0}
          for v in VOLUMES}
    unmapped = []

    def slot(v):
        return by.setdefault(
            v, {"valid": {}, "invalid": [], "rejected": [], "batches": 0})
    # Never assemble raw provider output.  The guarded runner writes a separate
    # accepted artifact containing only request-level responses that passed the
    # exact-ID, stop-reason, JSON, and usage checks.  This lets a large Batch
    # salvage its good requests without letting its failed neighbours leak into
    # delivery.
    accepted_paths = sorted(
        path for directory in _artifact_dirs(live)
        for path in directory.glob("*.accepted.jsonl"))
    for path in accepted_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("custom_id", "")
            vol = _volume_of(cid)
            if vol is None:
                # vocabtest is a deliberate None and is not a problem; anything
                # else is a paid response we are about to throw away, and the
                # caller must hear about it.
                if "vocabtest" not in (cid or ""):
                    unmapped.append(cid)
                continue
            slot(vol)["batches"] += 1
            resp = row.get("response") or {}
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if resp.get("status_code") != 200 or len(choices) != 1 \
                    or choices[0].get("finish_reason") != "stop":
                slot(vol)["invalid"].append(row.get("custom_id"))
                continue
            text = choices[0].get("message", {}).get("content")
            try:
                values, missing = parse_response(text, [], validate=True)
            except Exception:
                slot(vol)["invalid"].append(row.get("custom_id"))
                continue
            if missing:
                slot(vol)["invalid"].append(row.get("custom_id"))
                continue
            overlap = set(by[vol]["valid"]) & set(values)
            if overlap:
                # A repeated provider entry might hide a conflicting result; keep
                # the first provenance-bearing result and make the anomaly visible.
                slot(vol)["invalid"].append(
                    f"{row.get('custom_id')}: duplicate entries {sorted(overlap)[:3]}")
            slot(vol)["valid"].update({eid: value for eid, value in values.items()
                                      if eid not in overlap})
    # Accepted artifacts intentionally omit rejected requests. Read the guarded
    # validation reports as well so partial-batch failures remain visible in
    # CORPUS_SUMMARY instead of disappearing merely because salvage succeeded.
    validation_paths = sorted(
        path for directory in _artifact_dirs(live)
        for path in directory.glob("*.validation.json"))
    for path in validation_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rejected = report.get("rejected_custom_ids") or {}
        if not isinstance(rejected, dict):
            continue
        for cid, reason in rejected.items():
            vol = _volume_of(cid)
            if vol is None:
                if "vocabtest" not in (cid or ""):
                    unmapped.append(cid)
                continue
            item = f"{cid}: {reason}"
            if item not in slot(vol)["rejected"]:
                slot(vol)["rejected"].append(item)
    if unmapped:
        # These are PAID responses about to be discarded. Silence here is what
        # would have thrown away 701157 and 701179 after they were billed.
        print(f"WARNING: {len(unmapped)} accepted response(s) could not be mapped "
              f"to a volume and were NOT assembled, e.g. {unmapped[:3]}. Check the "
              f"custom_id convention before delivering.", file=sys.stderr)
    return by


def read_vocabtest_rows(live, tag: str = "701054-vocabtest"):
    """Read one isolated prompt-experiment result without mixing it into delivery.

    Experiment rows deliberately share source entry IDs with the delivered
    volume.  They therefore cannot use ``read_rows_by_volume``: that would
    either duplicate delivered records or lose the experiment during
    de-duplication.  The returned values are materialized into a distinct,
    provenance-bearing file for vocab_ab_report.py.
    """
    result = {"valid": {}, "invalid": [], "batches": 0}
    paths = sorted(path for directory in _artifact_dirs(live)
                   for path in directory.glob("*.accepted.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if tag not in (row.get("custom_id") or ""):
                continue
            result["batches"] += 1
            resp = row.get("response") or {}
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if resp.get("status_code") != 200 or len(choices) != 1 \
                    or choices[0].get("finish_reason") != "stop":
                result["invalid"].append(row.get("custom_id"))
                continue
            try:
                values, missing = parse_response(
                    choices[0].get("message", {}).get("content"), [], validate=True)
            except Exception:
                result["invalid"].append(row.get("custom_id"))
                continue
            if missing:
                result["invalid"].append(row.get("custom_id"))
                continue
            overlap = set(result["valid"]) & set(values)
            if overlap:
                result["invalid"].append(
                    f"{row.get('custom_id')}: duplicate entries {sorted(overlap)[:3]}")
            result["valid"].update({entry_id: value for entry_id, value in values.items()
                                    if entry_id not in overlap})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=Path, default=Path("production/luna_live"))
    ap.add_argument("--accepted-dir", type=Path, action="append",
                    help="directory containing guarded *.accepted.jsonl and "
                         "*.validation.json artifacts; repeat to combine runs "
                         "(default: --live directory)")
    ap.add_argument("--corpus", type=Path, action="append", dest="corpus_dirs",
                    help="directory containing *.segmented.json sources; repeat "
                         "for additional corpus directories (default: "
                         "production/corpus)")
    ap.add_argument("--keep-partials", action="store_true",
                    help="keep page-truncated (partial) records in the delivered "
                         "output. Default DROPS them per Daniel's 2026-07-22 "
                    "convention (his references omit trailing/incomplete "
                    "records). The deterministic source corpus is unchanged, "
                    "so this is a reversible delivery-layer choice.")
    ap.add_argument("--withdrawn", type=Path,
                    default=Path("production/luna_v3/withdrawn_records.json"),
                    help="quarantine file listing records withdrawn from the "
                         "delivered corpus. Re-applied on every assembly, "
                         "because assembly rebuilds from source and would "
                         "otherwise resurrect them.")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="materialize and report coverage only; skip the expensive "
                    "QA, identity, and graph refresh stages")
    args = ap.parse_args(argv)

    corpus_dirs = args.corpus_dirs or [Path("production/corpus")]
    corpus_paths = discover_corpora(corpus_dirs)
    if not corpus_paths:
        raise SystemExit(
            "REFUSING: no numeric *.segmented.json corpus sources found in "
            + ", ".join(str(path) for path in corpus_dirs))

    import materialize_luna_results as M
    import run_pipeline

    withdrawn_ids = set()
    if args.withdrawn and args.withdrawn.exists():
        _w = json.loads(args.withdrawn.read_text(encoding="utf-8")).get("withdrawn") or []
        withdrawn_ids = {str(r.get("id")) for r in _w if r.get("id")}
        print(f"withdrawal list: {len(withdrawn_ids)} record(s) held out "
              f"({args.withdrawn})")

    accepted_dirs = args.accepted_dir or [args.live]
    by = read_rows_by_volume(accepted_dirs)
    vocabtest = read_vocabtest_rows(accepted_dirs)
    provider_volumes = {
        vol for vol, rows in by.items()
        if rows["valid"] or rows["invalid"] or rows["batches"]
    }
    missing_sources = sorted(provider_volumes - set(corpus_paths))
    if missing_sources:
        raise ValueError(
            "accepted provider output has no supplied segmented source for "
            f"volume(s) {missing_sources}; add their directory with --corpus")
    outdir = args.live / "assembled"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"volumes": {}, "totals": {}}
    materialized_files = []
    tot_corpus = tot_mat = tot_missing = tot_invalid = 0

    for vol, corpus_path in sorted(corpus_paths.items()):
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        # Older/newer segmentation tools did not always persist a top-level
        # volume. The filename was validated by discover_corpora(), so inject
        # it here to prevent materialized outputs with ``volume: ""``.
        if not corpus.get("volume"):
            corpus["volume"] = vol
        volume_rows = by.setdefault(
            vol, {"valid": {}, "invalid": [], "rejected": [], "batches": 0})
        extracted = volume_rows["valid"]
        # A provider may return an identifier that is syntactically valid for
        # its request but not present in the current deterministic source (for
        # example after an upstream segmentation revision).  It must never be
        # materialized into the corpus, but it must remain visible in the QA
        # summary.  Missing source IDs are intentionally left untouched below
        # so materialize() reports them as incomplete coverage.
        corpus_ids = {str(entry.get("id")) for entry in corpus.get("entries", [])}
        extracted, unexpected_ids = filter_provider_only(volume_rows, corpus_ids)
        if not extracted:
            corpus_records = len(corpus.get("entries", []))
            summary["volumes"][vol] = {"state": "no provider output yet",
                                       "corpus_records": corpus_records,
                                       "materialized_records": 0,
                                       "missing_records": corpus_records,
                                       "invalid_batches": len(volume_rows["invalid"]),
                                       "rejected_requests": len(volume_rows["rejected"])}
            tot_corpus += corpus_records
            tot_missing += corpus_records
            continue
        result = M.materialize(corpus, extracted, allow_incomplete=True)
        # Daniel's convention (2026-07-22): drop page-truncated `partial` records
        # from the DELIVERED dataset; keep them only in the auditable source.
        result["entries"], dropped_partials = apply_delivery_convention(
            result["entries"], args.keep_partials)
        result["coverage"]["partials_dropped"] = dropped_partials

        # Withdrawn records must be re-withdrawn on every assembly.
        #
        # withdraw_records.py edits the materialized files in place, but assembly
        # rebuilds them from the batch output, so without this the next person to
        # run this script silently RESURRECTS two records whose text is the
        # transcription model apologising rather than the manuscript. A
        # withdrawal that only holds until the next rebuild is not a withdrawal.
        #
        # The quarantine file is the single source of truth for what is out and
        # why, so this reads it rather than keeping a second list in code.
        withdrawn_here = 0
        if withdrawn_ids:
            before = len(result["entries"])
            result["entries"] = [e for e in result["entries"]
                                 if str(e.get("id")) not in withdrawn_ids]
            withdrawn_here = before - len(result["entries"])
            if withdrawn_here:
                print(f"  {vol}: re-applied {withdrawn_here} withdrawal(s) "
                      f"from {args.withdrawn.name}")
        result["coverage"]["withdrawn_records"] = withdrawn_here
        result["coverage"]["materialized_records"] = len(result["entries"])
        mat_path = outdir / f"{vol}.materialized.json"
        mat_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        materialized_files.append((vol, mat_path))
        cov = result["coverage"]
        # per-volume QA/identity/graph
        pipe_dir = args.live / f"{vol}_final_pipeline"
        if not args.skip_pipeline:
            run_pipeline.main([str(mat_path), "--tag", vol, "--outdir", str(pipe_dir)])
        complete = cov["missing_records"] == 0
        state = ("COMPLETE" if not (by[vol]["invalid"] or by[vol]["rejected"])
                 else "COMPLETE_WITH_REPAIRED_ANOMALIES") if complete else "PARTIAL"
        summary["volumes"][vol] = {
            "state": state,
            "corpus_records": cov["corpus_records"],
            "materialized_records": cov["materialized_records"],
            "partials_dropped": dropped_partials,
            "missing_records": cov["missing_records"],
            "invalid_batches": len(by[vol]["invalid"]),
            "rejected_requests": len(by[vol]["rejected"]),
            "provider_only_ids": unexpected_ids,
            "pipeline": None if args.skip_pipeline else str(pipe_dir)}
        tot_corpus += cov["corpus_records"]; tot_mat += cov["materialized_records"]
        tot_missing += cov["missing_records"]; tot_invalid += len(by[vol]["invalid"])
        print(f"{vol}: {cov['materialized_records']} delivered "
              f"(dropped {dropped_partials} partials) of {cov['corpus_records']} corpus "
              f"({state}; missing {cov['missing_records']}, "
              f"invalid batches {len(by[vol]['invalid'])}, "
              f"historical rejected requests {len(by[vol]['rejected'])})")

    # cross-volume linkage (people linked ACROSS volumes) once >1 volume present
    if len(materialized_files) > 1 and not args.skip_pipeline:
        corpus_dir = args.live / "corpus_final_pipeline"
        run_pipeline.main([str(p) for _, p in materialized_files]
                          + ["--tag", "CORPUS", "--outdir", str(corpus_dir)])
        summary["corpus_pipeline"] = str(corpus_dir)

    # Materialize the 701054 vocabulary-prompt experiment separately.  It is
    # intentionally excluded from delivered volume assembly above because its
    # entry IDs overlap the baseline 701054 extraction.
    if vocabtest["valid"] or vocabtest["invalid"]:
        source_path = corpus_paths.get("701054")
        if source_path is None:
            raise ValueError(
                "701054 vocabtest output exists but no 701054 segmented source "
                "was supplied via --corpus")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_ids = {str(entry.get("id")) for entry in source.get("entries", [])}
        unexpected = sorted(set(vocabtest["valid"]) - source_ids)
        extracted = {entry_id: value for entry_id, value in vocabtest["valid"].items()
                     if entry_id in source_ids}
        experiment = {"state": "INVALID" if vocabtest["invalid"] else "READY",
                      "batches": vocabtest["batches"],
                      "valid_records": len(extracted),
                      "invalid_batches": vocabtest["invalid"],
                      "provider_only_ids": unexpected}
        if extracted:
            experiment_result = M.materialize(source, extracted, allow_incomplete=True)
            experiment_result["provenance"]["experiment"] = (
                "701054 vocabulary-aware prompt A/B test; excluded from delivered corpus")
            experiment_path = outdir / "701054-vocabtest.materialized.json"
            experiment_path.write_text(
                json.dumps(experiment_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            experiment["materialized_path"] = str(experiment_path)
        summary["vocabtest"] = experiment

    total_rejected = sum(len(rows["rejected"]) for rows in by.values())
    summary["totals"] = {"corpus_records": tot_corpus, "materialized_records": tot_mat,
                         "withdrawn_records": len(withdrawn_ids),
                         "missing_records": tot_missing, "invalid_batches": tot_invalid,
                         "rejected_requests": total_rejected,
                         "volumes_with_output": len(materialized_files)}
    (args.live / "CORPUS_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTOTAL materialized: {tot_mat}/{tot_corpus} records; "
          f"missing {tot_missing}; invalid batches {tot_invalid}")
    print(f"-> {args.live / 'CORPUS_SUMMARY.json'}")
    if tot_missing or tot_invalid:
        print("NOTE: see CORPUS_SUMMARY.json; missing records and historical invalid "
              "batches are reported for repair, never silently dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
