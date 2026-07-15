#!/usr/bin/env python3
"""run_batch.py — plan cache-ordered, single-pass batched extraction for a volume.

    python run_batch.py VOLUME.json [--batch 10] [--shots N]
        [--emit messages_batch1.json]   # write the first batch's ready-to-send messages

Reports the input-token reduction vs the current per-entry style and how many
normalization calls folding eliminates. With --emit it writes the messages array
for the first batch so you can drop it straight into your chat client. This tool
builds and prices requests; it does NOT call any API.
"""
import argparse
import json

from ssda_nlp_tools.batch_extract import build_messages, plan_batches, token_report


def _load_entries(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("examples") or d.get("entries") or []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume")
    ap.add_argument("--training", default="training_data.json")
    ap.add_argument("--instructions", default=None,
                    help="optional instructions.json (uses its raw text as one system msg)")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--emit", metavar="PATH", help="write the first batch's messages array")
    args = ap.parse_args(argv)

    entries = _load_entries(args.volume)
    examples = json.load(open(args.training, encoding="utf-8"))["examples"]
    if args.shots is not None:
        examples = examples[:args.shots]
    instructions = []
    if args.instructions:
        instructions = [{"text": open(args.instructions, encoding="utf-8").read()}]

    tr = token_report(entries, examples, instructions, batch_size=args.batch, shots=args.shots)
    print(f"volume entries:            {tr['entries']}")
    print(f"few-shot examples:         {tr['shots']}  (prefix {tr['prefix_tokens']} tokens)")
    print(f"batches @ {args.batch}/call:          {tr['batches']}")
    print(f"per-entry input tokens:    {tr['per_entry_input_tokens']:,}")
    print(f"batched input tokens:      {tr['batched_input_tokens']:,}")
    print(f"input reduction:           {tr['input_reduction_x']}x")
    print(f"normalization calls saved: {tr['separate_normalization_calls_saved']} (folded into extraction)")

    if args.emit:
        batches = plan_batches(entries, args.batch)
        msgs = build_messages(batches[0], examples, instructions)
        with open(args.emit, "w", encoding="utf-8") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=2)
        print(f"\nfirst-batch messages ({len(msgs)} turns) -> {args.emit}")
        print("send with e.g. client.chat.completions.create(model=..., messages=<this>, "
              "response_format={'type':'json_object'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
