#!/usr/bin/env python3
"""Safely submit, collect, and validate capped GPT-5.6 Luna Batch jobs.

The source compact batch files are read-only inputs.  This runner keeps every
provider artefact under ``production/luna_live`` and only sends a request when
``--confirm`` is supplied.  Each submitted Batch job receives a conservative
reservation in the local ledger before the request is made; a completed job is
settled only after every requested batch is present, stopped normally, and
contains parseable JSON for every requested record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "gpt-5.6-luna"
# Batch API prices, conservatively computed from provider-reported token usage.
BATCH_INPUT_PER_M = 0.50
BATCH_OUTPUT_PER_M = 3.00


def read_compact(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty compact batch file: {path}")
    header = json.loads(lines[0]).get("header")
    if not isinstance(header, dict):
        raise ValueError(f"missing header: {path}")
    rows = [json.loads(line) for line in lines[1:] if line.strip()]
    if not rows or any(not row.get("custom_id") for row in rows):
        raise ValueError(f"missing request rows/custom IDs: {path}")
    return header, rows


def expected_entries(row: dict) -> set[str]:
    content = row["tail_message"]["content"]
    try:
        supplied = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tail message is not JSON for {row.get('custom_id')}") from exc
    ids = {str(item["entry"]) for item in supplied.get("entries", [])}
    if not ids:
        raise ValueError(f"no entry IDs in {row.get('custom_id')}")
    return ids


def request_lines(header: dict, rows: list[dict]):
    prefix = header["prefix_messages"]
    reasoning = header.get("reasoning", "low")
    for row in rows:
        body = {
            "model": header.get("model", MODEL),
            "messages": prefix + [row["tail_message"]],
            "response_format": {"type": "json_object"},
            "reasoning_effort": reasoning,
            "max_completion_tokens": 9000,
        }
        yield {"custom_id": row["custom_id"], "method": "POST",
               "url": "/v1/chat/completions", "body": body}


def normal_id(value: str) -> str:
    """Normalize only the historical temporary prefix used by the first pilot."""
    return value.removeprefix("luna-production-")


def cost_from_usage(usage: dict) -> float:
    return (float(usage.get("prompt_tokens", 0)) * BATCH_INPUT_PER_M
            + float(usage.get("completion_tokens", 0)) * BATCH_OUTPUT_PER_M) / 1_000_000


def validate_output(rows: list[dict], response_rows: list[dict]) -> dict:
    requested = {row["custom_id"]: expected_entries(row) for row in rows}
    returned: dict[str, dict] = {}
    errors = []
    for response in response_rows:
        raw_id = response.get("custom_id", "")
        custom_id = normal_id(raw_id)
        if custom_id in returned:
            errors.append(f"duplicate response custom_id: {raw_id}")
            continue
        returned[custom_id] = response
    if set(returned) != set(requested):
        errors.append("requested/returned custom IDs differ")
    prompt = completion = 0
    for custom_id, expected in requested.items():
        response = returned.get(custom_id, {})
        body = response.get("response", {}).get("body", {})
        if response.get("response", {}).get("status_code") != 200:
            errors.append(f"{custom_id}: non-200 response")
            continue
        choices = body.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            errors.append(f"{custom_id}: missing normal stop")
            continue
        try:
            parsed = json.loads(choices[0]["message"]["content"])
            got = {str(item["entry"]) for item in parsed.get("results", [])}
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{custom_id}: invalid JSON result ({exc})")
            continue
        if got != expected:
            errors.append(f"{custom_id}: requested/result entry IDs differ")
        usage = body.get("usage", {})
        if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
            errors.append(f"{custom_id}: missing provider token usage")
        prompt += int(usage.get("prompt_tokens", 0))
        completion += int(usage.get("completion_tokens", 0))
    return {"valid": not errors, "errors": errors, "request_count": len(requested),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "confirmed_usd_conservative": round(cost_from_usage(
                {"prompt_tokens": prompt, "completion_tokens": completion}), 7)}


def load_ledger(path: Path, cap: float) -> dict:
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
    else:
        ledger = {"cap_usd": cap, "confirmed_usd": 0.0, "reserved_usd": 0.0, "jobs": []}
    if float(ledger.get("cap_usd", cap)) != cap:
        raise ValueError(f"ledger cap ${ledger.get('cap_usd')} differs from requested ${cap}")
    return ledger


def write_json(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def api(key: str, method: str, url: str, payload: bytes | None = None):
    request = urllib.request.Request(url, data=payload, method=method,
        headers={"Authorization": f"Bearer {key}"})
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"provider HTTP {exc.code}: {text}") from exc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_file", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("production/luna_live"))
    ap.add_argument("--cap-usd", type=float, default=20.0)
    ap.add_argument("--take", type=int, default=50, help="maximum compact requests to submit")
    ap.add_argument("--reservation-per-request", type=float, default=0.04)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--poll", metavar="BATCH_ID", help="download and validate an existing job")
    args = ap.parse_args(argv)
    if args.take < 1 or args.reservation_per_request <= 0:
        ap.error("--take and --reservation-per-request must be positive")
    ledger_path = args.outdir / "spend_ledger.json"
    ledger = load_ledger(ledger_path, args.cap_usd)
    header, all_rows = read_compact(args.batch_file)
    known = set()
    for item in ledger.get("jobs", []):
        if item.get("status") not in {"submitted", "validated", "settled_by_validation"}:
            continue
        known.add(normal_id(item.get("custom_id", "")))
        known.update(normal_id(value) for value in item.get("expected_custom_ids", []))
    rows = [row for row in all_rows if row["custom_id"] not in known][:args.take]
    if args.poll:
        if not args.confirm:
            print("DRY RUN: --poll would retrieve and validate the named provider job.")
            return 0
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        job = api(key, "GET", f"https://api.openai.com/v1/batches/{args.poll}")
        if job.get("status") != "completed" or not job.get("output_file_id"):
            print(f"provider job {args.poll}: {job.get('status')}; no settlement performed")
            return 0
        req = urllib.request.Request(f"https://api.openai.com/v1/files/{job['output_file_id']}/content",
            headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=120) as reply:
            output = [json.loads(line) for line in reply.read().decode("utf-8").splitlines() if line]
        selected = next((item for item in ledger["jobs"] if item.get("job_id") == args.poll), None)
        if not selected:
            raise RuntimeError("job is not present in local ledger; refusing ambiguous settlement")
        ids = {normal_id(x) for x in selected.get("expected_custom_ids", [])}
        lookup = {row["custom_id"]: row for row in all_rows}
        validation = validate_output([lookup[x] for x in ids], output)
        output_path = args.outdir / f"{args.poll}.output.jsonl"
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
                               encoding="utf-8")
        write_json(args.outdir / f"{args.poll}.validation.json", validation)
        if not validation["valid"]:
            print("INVALID: provider output was saved for review; ledger unchanged")
            return 2
        selected.update({"status": "validated", **validation})
        ledger["reserved_usd"] = round(float(ledger["reserved_usd"]) - float(selected["reserved_usd"]), 7)
        ledger["confirmed_usd"] = round(float(ledger["confirmed_usd"]) + validation["confirmed_usd_conservative"], 7)
        write_json(ledger_path, ledger)
        print(f"VALIDATED {args.poll}: {validation['request_count']} requests, ${validation['confirmed_usd_conservative']:.6f}")
        return 0
    if not rows:
        print("No unsent compact requests remain in this file.")
        return 0
    reservation = round(len(rows) * args.reservation_per_request, 7)
    remaining = args.cap_usd - float(ledger.get("confirmed_usd", 0)) - float(ledger.get("reserved_usd", 0))
    print(f"prepared {len(rows)} requests from {args.batch_file.name}; hard reservation ${reservation:.2f}; available ${remaining:.6f}")
    if reservation > remaining + 1e-9:
        print("REFUSING: reservation would exceed the cumulative hard cap.")
        return 2
    if not args.confirm:
        print("DRY RUN ONLY -- no network call and no key access.")
        return 0
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    volume = str(header.get("volume", args.batch_file.stem.split('.')[0]))
    payload_path = args.outdir / f"{volume}-next.payload.jsonl"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in request_lines(header, rows)), encoding="utf-8")
    # Upload uses multipart; use the standard library rather than retaining a key or SDK state.
    boundary = "----ssda-luna-batch-boundary"
    content = payload_path.read_bytes()
    multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                 f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{payload_path.name}"\r\nContent-Type: application/jsonl\r\n\r\n').encode() + content + f"\r\n--{boundary}--\r\n".encode()
    upload_req = urllib.request.Request("https://api.openai.com/v1/files", data=multipart, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(upload_req, timeout=120) as reply:
        uploaded = json.loads(reply.read().decode("utf-8"))
    created = api(key, "POST", "https://api.openai.com/v1/batches", json.dumps({
        "input_file_id": uploaded["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"}).encode())
    receipt = {"job_id": created["id"], "input_file_id": uploaded["id"], "volume": volume,
               "request_count": len(rows), "expected_custom_ids": [r["custom_id"] for r in rows],
               "reserved_usd": reservation, "status": "submitted"}
    ledger["reserved_usd"] = round(float(ledger.get("reserved_usd", 0)) + reservation, 7)
    ledger.setdefault("jobs", []).append(receipt)
    write_json(args.outdir / f"{created['id']}.receipt.json", receipt)
    write_json(ledger_path, ledger)
    print(f"SUBMITTED {created['id']}: {len(rows)} requests, reservation ${reservation:.2f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        raise SystemExit(2)
