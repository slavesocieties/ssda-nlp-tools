#!/usr/bin/env python3
"""Capped cross-provider extraction bake-off (dry-run by default).

This evaluates the held-out pages 0035-0044 against the same prompt and the
same four eight-entry batches for every provider.  It never reads keys from a
file: when confirmed, it reads only the relevant API key from the process
environment.  A persistent per-model worst-case reservation prevents resumed
request sequences from cumulatively exceeding the cap.  Reservations are kept
when a network failure leaves billing uncertain, rather than risking a repeat
request that exceeds the agreed spend limit.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from ssda_nlp_tools.batch_extract import build_messages, parse_response, plan_batches


MODELS = {
    # Standard (interactive) USD / million tokens.  The test is interactive so
    # that outputs and provider usage are available immediately; production may
    # later use each provider's asynchronous Batch API at its published rate.
    # "id" is the wire model id sent to the provider when it differs from the
    # friendly name (Anthropic's Haiku 4.5 requires the dated id).
    "gemini-2.5-flash": {"provider": "gemini", "input": 0.30, "output": 2.50, "key": "GEMINI_API_KEY"},
    "gemini-3.5-flash": {"provider": "gemini", "input": 1.50, "output": 9.00, "key": "GEMINI_API_KEY"},
    "gpt-5.4-mini": {"provider": "openai", "input": 0.75, "output": 4.50, "key": "OPENAI_API_KEY"},
    "gpt-5.6-luna": {"provider": "openai", "input": 1.00, "output": 6.00, "key": "OPENAI_API_KEY", "fixed_temperature": False},
    "claude-haiku-4-5": {"provider": "anthropic", "input": 1.00, "output": 5.00, "key": "ANTHROPIC_API_KEY",
                          "id": "claude-haiku-4-5-20251001"},
    "claude-sonnet-5": {"provider": "anthropic", "input": 2.00, "output": 10.00, "key": "ANTHROPIC_API_KEY", "fixed_temperature": False},
}


class ProviderRejected(RuntimeError):
    """The provider returned a definitive 4xx: the request was refused, so it
    was NOT billed and its spend reservation can be safely released. Network
    failures and 5xx responses stay ambiguous (the request may have been
    processed) and keep their reservation."""


def _request(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Provider error bodies describe model/parameter access failures. They
        # never contain our Authorization header; cap the saved/displayed text.
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        if 400 <= exc.code < 500:
            raise ProviderRejected(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _anthropic(messages, model, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
             for m in messages if m["role"] != "system"]
    body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": turns}
    if MODELS[model].get("fixed_temperature", True):
        body["temperature"] = 0
    return body


def _gemini(messages, model, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    return {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                                  "responseMimeType": "application/json"}}


def _call(cfg, messages, model, key, max_tokens):
    provider = cfg["provider"]
    # The wire id is what the provider expects; it differs from the friendly
    # name only when a provider requires a dated id (e.g. Anthropic's Haiku 4.5).
    wire = cfg.get("id", model)
    if provider == "openai":
        body = {"model": wire, "messages": messages,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"}}
        if cfg.get("fixed_temperature", True):
            body["temperature"] = 0
        raw = _request("https://api.openai.com/v1/chat/completions",
                       {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                       body)
        usage = raw.get("usage", {})
        return raw["choices"][0]["message"]["content"], int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    if provider == "anthropic":
        body = _anthropic(messages, model, max_tokens)
        body["model"] = wire
        raw = _request("https://api.anthropic.com/v1/messages",
                       {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                       body)
        usage = raw.get("usage", {})
        text = "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text")
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    raw = _request(f"https://generativelanguage.googleapis.com/v1beta/models/{wire}:generateContent",
                   {"x-goog-api-key": key, "Content-Type": "application/json"},
                   _gemini(messages, model, max_tokens))
    usage = raw.get("usageMetadata", {})
    text = "".join(p.get("text", "") for p in raw.get("candidates", [{}])[0].get("content", {}).get("parts", []))
    return text, int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0))


def _load(root, heldout):
    examples = json.loads((root / "training_data.json").read_text(encoding="utf-8"))["examples"]
    source = json.loads((root / heldout).read_text(encoding="utf-8"))["examples"]
    entries = [{"entry": e["entry"], "raw": e.get("raw") or e.get("normalized") or ""} for e in source]
    return examples, entries


def _input_token_ceiling(messages):
    """A deliberately high ceiling for provider input-token accounting.

    Token estimators can undercount an unfamiliar provider's tokenizer. UTF-8
    bytes plus a small per-message allowance is intentionally conservative for
    this fixed text-only prompt and is used only for spend reservations.
    """
    return sum(len(m["content"].encode("utf-8")) + 128 for m in messages)


def _read_ledger(path):
    """Return the cumulative confirmed and conservatively reserved spend."""
    if not path.exists():
        return {"models": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot safely read spend ledger {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
        raise RuntimeError(f"Spend ledger {path} has an invalid format.")
    return data


def _ledger_amounts(ledger, model):
    row = ledger.setdefault("models", {}).setdefault(model, {})
    confirmed = float(row.get("confirmed_usd", 0.0))
    reserved = float(row.get("reserved_usd", 0.0))
    if confirmed < 0 or reserved < 0:
        raise RuntimeError(f"Spend ledger has negative values for {model}.")
    return confirmed, reserved


def _write_ledger(path, ledger):
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reserve(ledger, model, amount):
    confirmed, reserved = _ledger_amounts(ledger, model)
    ledger["models"][model] = {"confirmed_usd": confirmed, "reserved_usd": reserved + amount}


def _settle(ledger, model, reserved_amount, confirmed_amount):
    """Convert one completed request's reservation to known provider usage."""
    confirmed, reserved = _ledger_amounts(ledger, model)
    if reserved + 1e-9 < reserved_amount:
        raise RuntimeError(f"Spend ledger reservation for {model} is inconsistent.")
    ledger["models"][model] = {
        "confirmed_usd": confirmed + confirmed_amount,
        "reserved_usd": max(0.0, reserved - reserved_amount),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--heldout", default="Sample_output/Generated_0035_0044_4o_prompt_V2.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--start-batch", type=int, default=0,
                    help="zero-based batch offset; permits short, resumable live runs")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="limit batches after --start-batch (use 1 for a single capped call)")
    ap.add_argument("--max-output-tokens", type=int, default=10000)
    ap.add_argument("--max-usd-per-model", type=float, default=1.00)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", default="model_bakeoff_results.json")
    ap.add_argument("--ledger", default="model_bakeoff_spend_ledger.json",
                    help="persistent spend ledger; reservations survive interrupted runs")
    args = ap.parse_args(argv)
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        ap.error(f"unknown model(s): {', '.join(unknown)}")
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        ap.error("--max-batches must be positive when supplied")
    if args.max_output_tokens <= 0:
        ap.error("--max-output-tokens must be positive")
    if args.max_usd_per_model <= 0:
        ap.error("--max-usd-per-model must be positive")

    root = Path(__file__).resolve().parent
    examples, entries = _load(root, args.heldout)
    all_batches = plan_batches(entries, args.batch_size)
    if args.start_batch < 0 or args.start_batch >= len(all_batches):
        ap.error(f"--start-batch must be between 0 and {len(all_batches) - 1}")
    batches = all_batches[args.start_batch:]
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
    prepared = [build_messages(batch, examples, []) for batch in batches]
    batch_maxima = {}
    tested_entries = sum(len(b) for b in batches)
    print(f"held-out entries in this run: {tested_entries}; requests/model: {len(prepared)}; few-shot examples: {len(examples)}")
    plans = {}
    for model in args.models:
        cfg = MODELS[model]
        maxima = [(_input_token_ceiling(messages) * cfg["input"]
                   + args.max_output_tokens * cfg["output"]) / 1e6 for messages in prepared]
        batch_maxima[model] = maxima
        maximum = sum(maxima)
        plans[model] = maximum
        print(f"{model}: worst-case ${maximum:.4f} (cap ${args.max_usd_per_model:.2f})")
        if maximum > args.max_usd_per_model:
            print(f"REFUSING {model}: worst-case cost exceeds the cap.")
    if not args.confirm:
        print("DRY RUN ONLY — no network call and no key access.")
        return 0
    if any(v > args.max_usd_per_model for v in plans.values()):
        return 2

    ledger_path = Path(args.ledger)
    try:
        ledger = _read_ledger(ledger_path)
    except RuntimeError as exc:
        print(f"REFUSING live run: {exc}")
        return 2
    eligible = []
    for model in args.models:
        cfg = MODELS[model]
        key = os.environ.get(cfg["key"])
        if not key:
            continue
        confirmed, reserved = _ledger_amounts(ledger, model)
        total_after_reservation = confirmed + reserved + plans[model]
        if total_after_reservation > args.max_usd_per_model + 1e-9:
            print(f"REFUSING {model}: cumulative confirmed/reserved spend would exceed the cap.")
        else:
            eligible.append(model)
    if len(eligible) != sum(bool(os.environ.get(MODELS[m]["key"])) for m in args.models):
        return 2
    # Reservations are made PER BATCH, immediately before each request (see the
    # batch loop). A reservation covering requests that were never sent taught
    # us the hard way (2026-07-16): an interrupt then strands the model's whole
    # budget. Per-batch granularity keeps the same guarantee — an interrupted
    # run can never silently double-spend — while limiting a stranded
    # reservation to the single request that was actually in flight.

    report = {"heldout": args.heldout, "entries": tested_entries, "batch_size": args.batch_size,
              "start_batch": args.start_batch,
              "max_usd_per_model": args.max_usd_per_model, "ledger": str(ledger_path), "models": {}}
    for model in args.models:
        cfg = MODELS[model]
        key = os.environ.get(cfg["key"])
        if not key:
            print(f"SKIP {model}: {cfg['key']} is not set.")
            report["models"][model] = {"status": "skipped", "reason": f"{cfg['key']} not set"}
            continue
        total_in = total_out = 0
        parsed_total = missing_total = 0
        rows = []
        unresolved_reservation = False
        for index, (batch, messages) in enumerate(zip(batches, prepared), 1):
            started = time.time()
            # Reserve exactly this request's worst case before sending it.
            _reserve(ledger, model, batch_maxima[model][index - 1])
            _write_ledger(ledger_path, ledger)
            try:
                text, inp, out = _call(cfg, messages, model, key, args.max_output_tokens)
            except ProviderRejected as exc:
                # A definitive 4xx (bad model id, malformed request, auth): the
                # provider refused it, so it was NOT billed. Release exactly
                # this request's reservation so a corrected re-run is not
                # blocked by phantom spend. Reservations stranded by OTHER
                # interrupted runs are deliberately left untouched.
                print(f"ERROR {model} batch {index}: {exc}")
                rows.append({"batch": index, "error": str(exc)})
                _settle(ledger, model, batch_maxima[model][index - 1], 0.0)
                _write_ledger(ledger_path, ledger)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, KeyError, IndexError, ValueError) as exc:
                print(f"ERROR {model} batch {index}: {exc}")
                rows.append({"batch": index, "error": str(exc)})
                # The provider may have received this request even though the
                # response was lost. Leave this batch's reservation in the
                # ledger until billing can be reviewed against the provider's
                # usage dashboard.
                unresolved_reservation = True
                break
            expected = [e["entry"] for e in batch]
            try:
                parsed, missing = parse_response(text, expected)
                parse_error = None
            except json.JSONDecodeError as exc:
                # A truncated JSON object is an observable model failure, not a
                # reason to lose the already-billed usage record.
                parsed, missing, parse_error = {}, expected, str(exc)
            total_in += inp; total_out += out; parsed_total += len(parsed); missing_total += len(missing)
            cost = (inp * cfg["input"] + out * cfg["output"]) / 1e6
            _settle(ledger, model, batch_maxima[model][index - 1], cost)
            _write_ledger(ledger_path, ledger)
            print(f"{model} [{index}/{len(prepared)}] {time.time()-started:.1f}s input={inp} output={out} ${cost:.4f} parsed={len(parsed)}/{len(expected)}")
            rows.append({"batch": index, "input_tokens": inp, "output_tokens": out, "usd": cost,
                         "parsed": len(parsed), "missing": missing, "parse_error": parse_error,
                         # Keep parsed records so run_eval / run_qa can score
                         # the provider outputs afterwards.  API keys and raw
                         # request headers are deliberately never written.
                         "results": [{"entry": eid, **value} for eid, value in parsed.items()]})
        total = (total_in * cfg["input"] + total_out * cfg["output"]) / 1e6
        confirmed, reserved = _ledger_amounts(ledger, model)
        report["models"][model] = {"status": "completed" if not unresolved_reservation else "billing_review_required",
                                    "input_tokens": total_in,
                                    "output_tokens": total_out, "usd": total,
                                    "usd_per_entry": total / max(1, tested_entries),
                                    "parsed": parsed_total, "missing": missing_total, "batches": rows,
                                    "ledger_confirmed_usd": confirmed,
                                    "ledger_reserved_usd": reserved}
        print(f"TOTAL {model}: ${total:.4f}; ${total/max(1,tested_entries):.5f}/entry; parsed {parsed_total}/{tested_entries}")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
