#!/usr/bin/env python3
"""poll_batch_jobs.py — check the three pending async Batch API jobs and save
their results + provider-reported usage once they finish.

    $env:GEMINI_API_KEY="..." ; $env:ANTHROPIC_API_KEY="..."
    python poll_batch_jobs.py

Polling is read-only (GET requests): it costs nothing and cannot resubmit
work. Keys are read from the environment only and are never printed or saved.
Job receipts read: gemini-3.5-flash_batch_job.json, sonnet_batch_job.json,
sonnet_cached_batch_job.json (skipped silently if absent).

The provider usage is saved for offline reconciliation. This script does not
modify model_bakeoff_spend_ledger.json because pricing changes over time and a
job receipt may omit auxiliary synchronous calls such as cache warm-ups.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _get(url, headers):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return json.dumps({"error": f"HTTP {exc.code}: "
                          f"{exc.read().decode('utf-8', errors='replace')[:400]}"})


def poll_gemini(receipt):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"status": "skipped", "reason": "GEMINI_API_KEY not set"}
    name = json.loads(receipt.read_text(encoding="utf-8"))["name"]
    raw = json.loads(_get(f"https://generativelanguage.googleapis.com/v1beta/{name}",
                          {"x-goog-api-key": key}))
    state = (raw.get("metadata", {}) or {}).get("state") or raw.get("state") or raw.get("error")
    out = {"job": name, "state": state}
    responses = (raw.get("response", {}) or {}).get("inlinedResponses", {})
    if responses:
        out["results"] = responses
    return out


def poll_anthropic(receipt):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"status": "skipped", "reason": "ANTHROPIC_API_KEY not set"}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    job_id = json.loads(receipt.read_text(encoding="utf-8"))["id"]
    raw = json.loads(_get(f"https://api.anthropic.com/v1/messages/batches/{job_id}", headers))
    out = {"job": job_id, "state": raw.get("processing_status"),
           "counts": raw.get("request_counts")}
    if raw.get("results_url"):
        # results_url is JSONL: one line per request with content + usage
        lines = _get(raw["results_url"], headers).splitlines()
        out["results"] = [json.loads(line) for line in lines if line.strip()]
        usage = [((r.get("result", {}) or {}).get("message", {}) or {}).get("usage", {})
                 for r in out["results"]]
        out["usage"] = usage
    return out


def main():
    jobs = [
        ("gemini-3.5-flash_batch_job.json", poll_gemini),
        ("sonnet_batch_job.json", poll_anthropic),
        ("sonnet_cached_batch_job.json", poll_anthropic),
    ]
    report = {}
    for name, poll in jobs:
        receipt = ROOT / name
        if not receipt.exists():
            continue
        result = poll(receipt)
        report[name] = result
        state = result.get("state") or result.get("status")
        done = "results" in result
        print(f"{name}: {state}" + ("  -> results retrieved" if done else ""))
    out = ROOT / "batch_jobs_status.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
