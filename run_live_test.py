#!/usr/bin/env python3
"""run_live_test.py — capped LIVE cost test: send a few real extraction batches
and measure the ACTUAL cost per image from the API's own usage numbers.

    # 1) see exactly what would be sent and what it should cost (no key needed)
    python run_live_test.py out_batches/239746.batches.jsonl --max-batches 3

    # 2) actually send it (requires OPENAI_API_KEY in the environment + --confirm)
    set OPENAI_API_KEY=sk-...        (PowerShell:  $env:OPENAI_API_KEY="sk-...")
    python run_live_test.py out_batches/239746.batches.jsonl --max-batches 3 --confirm

Safety rails (all hard, not advisory):
  * DRY-RUN BY DEFAULT — nothing is sent without --confirm.
  * --max-batches N   caps the number of requests (default 3).
  * --max-usd X       refuses to start if the PROJECTED spend exceeds X
                      (default $0.50), and stops mid-run if ACTUAL spend hits it.
  * The key is read from the environment only; it is never printed or stored.

Outputs: per-batch actual prompt/cached/completion tokens and cost, the measured
COST PER ENTRY and PER IMAGE, side-by-side with our offline projection, plus the
parsed extraction results (so quality can be eyeballed at the same time).
"""
import argparse
import json
import os
import sys
import time

from ssda_nlp_tools.batch_extract import parse_response, merge_with_faithful
from ssda_nlp_tools.cost import DEFAULT_PRICING, count_tokens


def load_compact_batches(path, max_batches):
    with open(path, encoding="utf-8") as fh:
        header = json.loads(fh.readline())["header"]
        rows = []
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= max_batches:
                break
    return header, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batchfile", help="a compact out_batches/<vol>.batches.jsonl")
    ap.add_argument("--max-batches", type=int, default=3)
    ap.add_argument("--max-usd", type=float, default=0.50,
                    help="hard spend cap (projected AND actual)")
    ap.add_argument("--model", default=None,
                    help="override the model id (default: from the batch header)")
    ap.add_argument("--confirm", action="store_true",
                    help="actually send; without this it is a dry run")
    ap.add_argument("--out", default="live_test_results.json")
    args = ap.parse_args(argv)

    header, rows = load_compact_batches(args.batchfile, args.max_batches)
    model = args.model or header["model"]
    price = DEFAULT_PRICING.get(model)

    # ---- projection (offline) ------------------------------------------------
    prefix_tok = sum(count_tokens(m["content"]) for m in header["prefix_messages"])
    n_entries = 0
    tail_tok = 0
    images = set()
    for r in rows:
        payload = json.loads(r["tail_message"]["content"])
        n_entries += len(payload["entries"])
        tail_tok += count_tokens(r["tail_message"]["content"])
        for e in payload["entries"]:
            images.add(e["entry"].rsplit("-", 1)[0])
    est_out = n_entries * 900
    if price:
        est_cost = ((prefix_tok + (len(rows) - 1) * prefix_tok * price.cached / price.input
                     + tail_tok) * price.input + est_out * price.output) / 1e6
    else:
        est_cost = float("nan")

    print(f"volume: {header['volume']}   model: {model}")
    print(f"batches: {len(rows)}   entries: {n_entries}   distinct images: {len(images)}")
    print(f"PROJECTED: ~{prefix_tok:,} prefix tok/call (cacheable) + {tail_tok:,} tail tok, "
          f"~{est_out:,} output tok -> ~${est_cost:.4f} total, "
          f"~${est_cost/max(1,len(images)):.5f}/image")

    if est_cost == est_cost and est_cost > args.max_usd:   # NaN-safe
        print(f"REFUSING: projection ${est_cost:.2f} exceeds --max-usd {args.max_usd}")
        return 2
    if not args.confirm:
        print("\nDRY RUN (no API call made). Re-run with --confirm and "
              "OPENAI_API_KEY set to send.")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in the environment. Set it and re-run.")
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai   (needed only for the live run)")
        return 2
    client = OpenAI()

    # ---- live run --------------------------------------------------------------
    totals = {"prompt": 0, "cached": 0, "completion": 0, "usd": 0.0}
    results, failures, merged_records = [], [], []
    for i, r in enumerate(rows, 1):
        messages = header["prefix_messages"] + [r["tail_message"]]
        payload = json.loads(r["tail_message"]["content"])
        ids = [e["entry"] for e in payload["entries"]]
        # the segmenter's FAITHFUL text, in canonical form, so we can keep both
        # faithful and normalized in the output (confirmed decision, 2026-07-16)
        canonical = [{"id": e["entry"], "images": [e["entry"].rsplit("-", 1)[0] + ".jpg"],
                     "text": e["transcription"]} for e in payload["entries"]]
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0,
            response_format={"type": "json_object"})
        dt = time.time() - t0
        u = resp.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        cost = 0.0
        if price:
            cost = ((u.prompt_tokens - cached) * price.input + cached * price.cached
                    + u.completion_tokens * price.output) / 1e6
        totals["prompt"] += u.prompt_tokens
        totals["cached"] += cached
        totals["completion"] += u.completion_tokens
        totals["usd"] += cost
        parsed, missing = parse_response(resp.choices[0].message.content, ids)
        print(f"  [{i}/{len(rows)}] {r['custom_id']}: {dt:.1f}s  "
              f"prompt={u.prompt_tokens:,} (cached {cached:,})  "
              f"out={u.completion_tokens:,}  ${cost:.4f}  "
              f"parsed {len(parsed)}/{len(ids)}"
              + (f"  MISSING {missing}" if missing else ""))
        results.append({"custom_id": r["custom_id"], "usage": {
            "prompt": u.prompt_tokens, "cached": cached,
            "completion": u.completion_tokens, "usd": cost},
            "parsed": parsed, "missing": missing})
        merged_records.extend(merge_with_faithful(canonical, parsed))
        failures.extend(missing)
        if totals["usd"] >= args.max_usd:
            print(f"  STOP: actual spend ${totals['usd']:.2f} reached --max-usd cap")
            break

    per_image = totals["usd"] / max(1, len(images))
    per_entry = totals["usd"] / max(1, n_entries)
    print("\n===== MEASURED =====")
    print(f"total: ${totals['usd']:.4f}  "
          f"(prompt {totals['prompt']:,} tok of which cached {totals['cached']:,}; "
          f"output {totals['completion']:,} tok)")
    print(f"ACTUAL cost/entry: ${per_entry:.5f}   ACTUAL cost/image: ${per_image:.5f}")
    print(f"vs projection:     ${est_cost/max(1,n_entries):.5f}/entry, "
          f"${est_cost/max(1,len(images)):.5f}/image")
    if totals["cached"] == 0 and len(rows) > 1:
        print("note: no cached tokens reported — cache may need >1024-token prefix "
              "and a few seconds between identical-prefix calls.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"volume": header["volume"], "model": model,
                   "totals": totals, "per_image_usd": per_image,
                   "per_entry_usd": per_entry, "results": results,
                   # confirmed 2026-07-16: keep BOTH faithful (what Archivault
                   # produced) and normalized (the LLM's cleaned-up version) —
                   # neither replaces the other in the final record
                   "records": merged_records},
                  fh, ensure_ascii=False, indent=1)
    print(f"-> {args.out}  ({len(merged_records)} records, each with "
          f"text_faithful + text_normalized + data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
