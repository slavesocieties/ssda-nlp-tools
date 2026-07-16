#!/usr/bin/env python3
"""run_corpus_prompts.py — turn segmented volumes into READY-TO-SEND extraction
batches (the only paid step of the pipeline), priced before you spend a cent.

    python run_corpus_prompts.py [--corpus out_corpus] [--outdir out_batches]
                                 [--batch 10] [--shots 15] [--model claude-haiku-4.5]
                                 [--volumes 100800 239746 ...] [--limit N]

For every out_corpus/<vol>.segmented.json this writes a COMPACT batch file
    out_batches/<vol>.batches.jsonl
        line 1: {"header": {"volume", "model", "prefix_messages": [...]}}
        rest:   {"custom_id": "<vol>-b0007", "tail_message": {...}}
(the 14k-token static prefix is identical for every call by design, so it is
stored ONCE per volume instead of ~60 KB repeated per line — ~1 GB saved across
the corpus). To produce the verbatim OpenAI **Batch API** upload file
(24h turnaround, 50% cheaper than interactive) for a volume:

    python run_corpus_prompts.py --expand out_batches/100800.batches.jsonl

The messages come from ssda_nlp_tools.batch_extract: cache-ordered
(static system+instructions+few-shots first, byte-identical across batches),
real few-shot pool from training_data.json, normalization folded into the same
call. Partial entries are included but tagged so the model knows the text may
be truncated. A manifest.json holds per-volume counts + token/cost projections.

No API calls are made — this PREPARES and PRICES the work.
"""
import argparse
import glob
import json
import os

from ssda_nlp_tools.batch_extract import build_messages, plan_batches
from ssda_nlp_tools.cost import DEFAULT_PRICING, count_tokens


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="out_corpus")
    ap.add_argument("--outdir", default="out_batches")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--shots", type=int, default=15)
    ap.add_argument("--model", default="claude-haiku-4.5")
    ap.add_argument("--training", default="training_data.json")
    ap.add_argument("--instructions", default=None)
    ap.add_argument("--volumes", nargs="*", default=None, help="only these volume ids")
    ap.add_argument("--limit", type=int, default=None, help="max volumes (smoke test)")
    ap.add_argument("--expand", metavar="BATCHFILE",
                    help="expand a compact batch file to verbatim Batch-API JSONL and exit")
    args = ap.parse_args(argv)

    if args.expand:
        out = args.expand.replace(".batches.jsonl", ".batchapi.jsonl")
        with open(args.expand, encoding="utf-8") as fh, \
             open(out, "w", encoding="utf-8") as oh:
            header = json.loads(fh.readline())["header"]
            n = 0
            for line in fh:
                row = json.loads(line)
                oh.write(json.dumps({
                    "custom_id": row["custom_id"],
                    "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": header["model"],
                             "messages": header["prefix_messages"] + [row["tail_message"]],
                             "response_format": {"type": "json_object"}},
                }, ensure_ascii=False) + "\n")
                n += 1
        print(f"expanded {n} requests -> {out}  (upload this to the Batch API)")
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    examples = json.load(open(args.training, encoding="utf-8"))["examples"][: args.shots]
    instructions = ([{"text": open(args.instructions, encoding="utf-8").read()}]
                    if args.instructions else [])

    files = sorted(glob.glob(os.path.join(args.corpus, "*.segmented.json")))
    if args.volumes:
        want = set(args.volumes)
        files = [f for f in files
                 if os.path.basename(f).split(".")[0] in want]
    if args.limit:
        files = files[: args.limit]

    price = DEFAULT_PRICING.get(args.model)
    manifest = {"model": args.model, "batch_size": args.batch, "shots": len(examples),
                "volumes": {}}
    tot_calls = tot_entries = tot_partial = 0
    tot_in_tokens = tot_prefix = 0

    prefix_tokens = None
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        vol = str(d.get("volume") or os.path.basename(f).split(".")[0])
        entries = []
        n_partial = 0
        for e in d.get("entries", []):
            text = e.get("text", "")
            if not text.strip():
                continue
            if e.get("partial"):
                n_partial += 1
                # keep it — tag so the model knows the tail may be missing
                text += "\n[NOTE: entry may be truncated at a page/volume boundary]"
            entries.append({"entry": e.get("id", ""), "raw": text})
        if not entries:
            continue

        batches = plan_batches(entries, args.batch)
        out_path = os.path.join(args.outdir, f"{vol}.batches.jsonl")
        vol_tail_tokens = 0
        with open(out_path, "w", encoding="utf-8") as fh:
            for bi, b in enumerate(batches):
                msgs = build_messages(b, examples, instructions)
                if prefix_tokens is None:   # identical for every batch by design
                    prefix_tokens = sum(count_tokens(m["content"]) for m in msgs[:-1])
                if bi == 0:                 # store the shared prefix once
                    fh.write(json.dumps({"header": {
                        "volume": vol, "model": args.model,
                        "prefix_messages": msgs[:-1]}}, ensure_ascii=False) + "\n")
                vol_tail_tokens += count_tokens(msgs[-1]["content"])
                fh.write(json.dumps({"custom_id": f"{vol}-b{bi:04d}",
                                     "tail_message": msgs[-1]},
                                    ensure_ascii=False) + "\n")

        n_calls = len(batches)
        vol_in = n_calls * (prefix_tokens or 0) + vol_tail_tokens
        manifest["volumes"][vol] = {
            "entries": len(entries), "partial_tagged": n_partial,
            "calls": n_calls, "input_tokens_nocache": vol_in,
            "file": os.path.basename(out_path),
        }
        tot_calls += n_calls; tot_entries += len(entries); tot_partial += n_partial
        tot_in_tokens += vol_in; tot_prefix += n_calls * (prefix_tokens or 0)

    # cost projection: interactive-cached vs Batch API (50% off, prefix still cached)
    est = {}
    if price and tot_calls:
        out_tokens = tot_entries * 900          # data json + folded normalization
        cached_in = (prefix_tokens or 0) * len(manifest["volumes"]) \
            + (tot_prefix - (prefix_tokens or 0) * len(manifest["volumes"])) * (price.cached / price.input) \
            + (tot_in_tokens - tot_prefix)
        interactive = (cached_in * price.input + out_tokens * price.output) / 1e6
        batch_api = 0.5 * ((tot_in_tokens - tot_prefix * 0.9) * price.input
                           + out_tokens * price.output) / 1e6
        est = {"interactive_cached_usd": round(interactive, 2),
               "batch_api_usd": round(batch_api, 2),
               "est_output_tokens": out_tokens}
    manifest["totals"] = {"volumes": len(manifest["volumes"]), "entries": tot_entries,
                          "partial_tagged": tot_partial, "calls": tot_calls,
                          "prefix_tokens_per_call": prefix_tokens,
                          "input_tokens_nocache": tot_in_tokens, **est}
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    t = manifest["totals"]
    print(f"volumes:  {t['volumes']}   entries: {t['entries']:,} "
          f"({t['partial_tagged']:,} tagged partial)")
    print(f"LLM calls: {t['calls']:,}  (batch={args.batch}, shots={len(examples)}, "
          f"prefix {prefix_tokens:,} tok/call — cacheable)")
    if est:
        print(f"projected extraction cost [{args.model}]: "
              f"~${est['interactive_cached_usd']:,} interactive w/ cache, "
              f"~${est['batch_api_usd']:,} via Batch API")
    print(f"-> {args.outdir}/<vol>.batches.jsonl + manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
