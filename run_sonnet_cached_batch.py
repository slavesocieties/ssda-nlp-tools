#!/usr/bin/env python3
"""Run a deliberately capped Claude Sonnet 5 cached Batch API evaluation.

The command is dry-run by default. A confirmed run first warms a one-hour
prompt cache containing the fixed 15-shot prefix, verifies that Anthropic
reports cache creation, and only then submits two four-entry Batch requests.
No API key is read from a file or written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from ssda_nlp_tools.batch_extract import build_messages, plan_batches


MODEL = "claude-sonnet-5"
INPUT_BATCH = 1.00
OUTPUT_BATCH = 5.00
CACHE_WRITE_1H_STANDARD = 4.00
CACHE_READ_BATCH = 0.10


def _post(url, key, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"Anthropic returned HTTP {exc.code}: {detail}") from exc


def _schema():
    # Anthropic's grammar requires closed objects. This mirrors every field in
    # the local training examples while keeping historically absent fields
    # optional.
    relation = {
        "type": "object",
        "properties": {
            "related_person": {"type": "string"},
            "relationship_type": {"type": "string"},
        },
        "required": ["related_person", "relationship_type"],
        "additionalProperties": False,
    }
    person = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "age": {"type": "string"},
            "ethnicity": {"type": "string"},
            "free": {"type": "boolean"},
            "legitimate": {"type": "boolean"},
            "occupation": {"type": "string"},
            "origin": {"type": "string"},
            "phenotype": {"type": "string"},
            "rank": {"type": "string"},
            "titles": {"type": "array", "items": {"type": "string"}},
            "relationships": {"type": "array", "items": relation},
        },
        "required": ["id", "name"],
        "additionalProperties": False,
    }
    event = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "principals": {"type": "array", "items": {"type": "string"}},
            "date": {"type": "string"},
        },
        "required": ["type", "principals"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entry": {"type": "string"},
                        "normalized": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "people": {"type": "array", "items": person},
                                "events": {"type": "array", "items": event},
                            },
                            "required": ["people", "events"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["entry", "normalized", "data"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def _split_messages(messages):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    prefix, tail = turns[:-1], turns[-1]
    rendered = []
    for index, turn in enumerate(prefix):
        content = turn["content"]
        if index == len(prefix) - 1:
            content = [{"type": "text", "text": content,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        rendered.append({
            "role": "assistant" if turn["role"] == "assistant" else "user",
            "content": content,
        })
    return system, rendered, tail["content"]


def _token_ceiling(*texts):
    return sum(len(text.encode("utf-8")) + 128 for text in texts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--max-usd", type=float, default=1.00)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--out", default="sonnet_cached_batch_job.json")
    args = parser.parse_args(argv)
    if args.max_usd <= 0 or args.max_output_tokens <= 0:
        parser.error("--max-usd and --max-output-tokens must be positive")

    root = Path(__file__).resolve().parent
    examples = json.loads((root / "training_data.json").read_text(encoding="utf-8"))["examples"]
    source = json.loads((root / "Sample_output/Generated_0035_0044_4o_prompt_V2.json").read_text(encoding="utf-8"))["examples"]
    entries = [{"entry": e["entry"], "raw": e.get("raw") or e.get("normalized") or ""}
               for e in source[:8]]
    batches = plan_batches(entries, 4)
    rendered = [_split_messages(build_messages(batch, examples, [])) for batch in batches]
    system, prefix, _ = rendered[0]
    prefix_texts = [system] + [
        block["content"][0]["text"] if isinstance(block["content"], list) else block["content"]
        for block in prefix
    ]
    prefix_ceiling = _token_ceiling(*prefix_texts)
    tail_ceiling = sum(_token_ceiling(tail) for _, _, tail in rendered)
    maximum = (
        prefix_ceiling * CACHE_WRITE_1H_STANDARD
        + prefix_ceiling * len(batches) * CACHE_READ_BATCH
        + tail_ceiling * INPUT_BATCH
        + len(batches) * args.max_output_tokens * OUTPUT_BATCH
    ) / 1_000_000
    print(f"entries: {len(entries)}; batch requests: {len(batches)}; max output/request: {args.max_output_tokens}")
    print(f"conservative maximum USD: {maximum:.4f}; cap USD: {args.max_usd:.2f}")
    if maximum > args.max_usd:
        print("REFUSING: conservative maximum exceeds the cap.")
        return 2
    if not args.confirm:
        print("DRY RUN ONLY — no key access or network call.")
        return 0

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("REFUSING: ANTHROPIC_API_KEY is not set.")
        return 2
    warmup = {
        "model": MODEL,
        "max_tokens": 1,
        "system": system,
        "messages": prefix + [{"role": "user", "content": "Cache warm-up. Reply with a period."}],
    }
    warm = _post("https://api.anthropic.com/v1/messages", key, warmup)
    usage = warm.get("usage", {})
    cache_created = int(usage.get("cache_creation_input_tokens", 0))
    cache_read = int(usage.get("cache_read_input_tokens", 0))
    if cache_created <= 0 and cache_read <= 0:
        raise RuntimeError("Cache was neither created nor read; Batch requests were not submitted.")
    params = []
    for index, (_, turns, tail) in enumerate(rendered, 1):
        params.append({
            "custom_id": f"sonnet-cached-heldout-{index}",
            "params": {
                "model": MODEL,
                "max_tokens": args.max_output_tokens,
                "system": system,
                "messages": turns + [{"role": "user", "content": tail}],
                "output_config": {"format": {"type": "json_schema", "schema": _schema()}},
            },
        })
    job = _post("https://api.anthropic.com/v1/messages/batches", key, {"requests": params})
    record = {
        "id": job.get("id"),
        "processing_status": job.get("processing_status"),
        "cache_creation_input_tokens": cache_created,
        "cache_read_input_tokens": cache_read,
        "conservative_max_usd": maximum,
    }
    (root / args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
