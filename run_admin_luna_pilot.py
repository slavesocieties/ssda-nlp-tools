#!/usr/bin/env python3
"""Capped GPT-5.6 Luna extraction of administrative dossiers (dry-run by default).

The faithful transcription remains local in the dossier file.  This runner asks
Luna only for auditable structured metadata and stores no API key.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "gpt-5.6-luna"
INPUT_RATE = 1.00
OUTPUT_RATE = 6.00
SYSTEM = """You extract structured metadata from Spanish colonial administrative archival dossiers.
Return JSON only. Preserve uncertainty: omit facts that the dossier does not support.
Do not infer family, enslavement, or sacramental events from this administrative material.
Use source-language names and short supporting evidence quotes.
This is metadata extraction, not transcription: never reproduce a page or a
paragraph. Keep every evidence quote to 12 words or fewer. Per extraction unit
return at most 3 document types, 5 organizations, 8 people, 5 places, 6 dates,
8 actions, and 3 uncertainties. Omit lower-value repetitions.

Return exactly:
{
  "document_id": "...",
  "document_types": ["petition"],
  "organizations": [{"name":"...", "role":"...", "evidence":"..."}],
  "people": [{"name":"...", "roles":["..."], "evidence":"..."}],
  "places": [{"name":"...", "evidence":"..."}],
  "dates": [{"text":"...", "iso_date":null, "evidence":"..."}],
  "actions": [{"type":"petition|approval|denial|appointment|regulation|other", "actors":["..."], "target":"...", "evidence":"..."}],
  "uncertainties": ["..."]
}
All array fields are required, even when empty.  `iso_date` must be ISO-8601 only when fully stated; otherwise null."""


class ProviderRejected(RuntimeError):
    pass


def _load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data.get("documents")
    if not isinstance(docs, list) or not docs:
        raise RuntimeError("Input has no administrative dossiers.")
    return docs


def _messages(doc):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({
            "document_id": doc["id"], "title": doc["title"],
            "metadata": doc.get("metadata", {}), "faithful_text": doc["faithful_text"],
        }, ensure_ascii=False)},
    ]


def _chunk(doc, size):
    """Split long dossiers by source pages while preserving provenance."""
    if not size or len(doc["pages"]) <= size:
        return [doc]
    chunks = []
    for start in range(0, len(doc["pages"]), size):
        pages = doc["pages"][start:start + size]
        end = start + len(pages)
        clone = dict(doc)
        clone["id"] = f"{doc['id']}--p{start + 1:02d}-{end:02d}"
        clone["title"] = f"{doc['title']} (pages {start + 1}-{end})"
        clone["pages"] = pages
        clone["source_images"] = [p["file"] for p in pages]
        clone["faithful_text"] = "\n\n".join(
            f"[source image: {p['file']}]\n{p['transcription']}".strip() for p in pages)
        clone["parent_document_id"] = doc["id"]
        chunks.append(clone)
    return chunks


def _ceiling(messages, max_output):
    # UTF-8 bytes deliberately over-reserve unfamiliar tokenizer behavior.
    prompt = sum(len(m["content"].encode("utf-8")) + 128 for m in messages)
    return (prompt * INPUT_RATE + max_output * OUTPUT_RATE) / 1_000_000


def _read_ledger(path):
    if not path.exists():
        return {"confirmed_usd": 0.0, "reserved_usd": 0.0}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"confirmed_usd": float(data.get("confirmed_usd", 0.0)),
            "reserved_usd": float(data.get("reserved_usd", 0.0))}


def _write_ledger(path, ledger):
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def _request_body(messages, max_output, reasoning_effort):
    body = {"model": MODEL, "messages": messages, "max_completion_tokens": max_output,
            "response_format": {"type": "json_object"},
            "reasoning_effort": reasoning_effort}
    return body


def _call(key, messages, max_output, reasoning_effort):
    body = _request_body(messages, max_output, reasoning_effort)
    request = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if 400 <= exc.code < 500:
            raise ProviderRejected(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    usage = raw.get("usage", {})
    choice = raw["choices"][0]
    return (choice["message"]["content"], choice.get("finish_reason"),
            int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))


def _valid(data, doc_id):
    required = ("document_types", "organizations", "people", "places", "dates", "actions", "uncertainties")
    return (isinstance(data, dict) and data.get("document_id") == doc_id
            and all(isinstance(data.get(key), list) for key in required))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--out", default="drive_pilots/3952_luna_results.json")
    ap.add_argument("--ledger", default="drive_pilots/3952_luna_spend_ledger.json")
    ap.add_argument("--max-output-tokens", type=int, default=5000)
    ap.add_argument("--max-usd", type=float, default=0.25)
    ap.add_argument("--only-documents", nargs="*", default=None,
                    help="source document IDs to run; omit for all")
    ap.add_argument("--page-chunk-size", type=int, default=0,
                    help="split selected dossiers into chunks of this many pages")
    ap.add_argument("--skip-chunk-ids", nargs="*", default=(),
                    help="already-completed chunk IDs to exclude after page splitting")
    ap.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                    default="none", help="reasoning budget; none is appropriate for bounded extraction")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args(argv)
    docs = _load(Path(args.input))
    if args.only_documents is not None:
        wanted = set(args.only_documents)
        docs = [doc for doc in docs if doc["id"] in wanted]
        if len(docs) != len(wanted):
            ap.error("one or more --only-documents IDs were not found")
    docs = [chunk for doc in docs for chunk in _chunk(doc, args.page_chunk_size)]
    if args.skip_chunk_ids:
        skipped = set(args.skip_chunk_ids)
        docs = [doc for doc in docs if doc["id"] not in skipped]
        if not docs:
            ap.error("all selected chunks were skipped")
    prepared = [(doc, _messages(doc)) for doc in docs]
    ceiling = sum(_ceiling(messages, args.max_output_tokens) for _, messages in prepared)
    print(f"dossiers: {len(docs)}; max output/dossier: {args.max_output_tokens}")
    print(f"worst-case new spend: ${ceiling:.4f}; cap: ${args.max_usd:.2f}")
    if ceiling > args.max_usd:
        print("REFUSING: worst-case spend exceeds cap.")
        return 2
    if not args.confirm:
        print("DRY RUN ONLY — no network call and no key access.")
        return 0
    ledger_path = Path(args.ledger)
    ledger = _read_ledger(ledger_path)
    if ledger["confirmed_usd"] + ledger["reserved_usd"] + ceiling > args.max_usd + 1e-9:
        print("REFUSING: cumulative confirmed/reserved spend would exceed cap.")
        return 2
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("REFUSING: OPENAI_API_KEY is not set.")
        return 2
    rows = []
    for number, (doc, messages) in enumerate(prepared, 1):
        reserved = _ceiling(messages, args.max_output_tokens)
        ledger["reserved_usd"] += reserved
        _write_ledger(ledger_path, ledger)
        started = time.time()
        try:
            text, stop, inp, out = _call(key, messages, args.max_output_tokens, args.reasoning_effort)
        except ProviderRejected as exc:
            ledger["reserved_usd"] -= reserved
            _write_ledger(ledger_path, ledger)
            print(f"ERROR {doc['id']}: {exc}")
            break
        except Exception as exc:
            print(f"ERROR {doc['id']}: {exc}; reservation retained for billing review")
            break
        cost = (inp * INPUT_RATE + out * OUTPUT_RATE) / 1_000_000
        ledger["reserved_usd"] -= reserved
        ledger["confirmed_usd"] += cost
        _write_ledger(ledger_path, ledger)
        try:
            parsed = json.loads(text)
            valid = _valid(parsed, doc["id"])
            error = None if valid else "schema or document_id validation failed"
        except json.JSONDecodeError as exc:
            parsed, valid, error = None, False, str(exc)
        rows.append({"document_id": doc["id"], "title": doc["title"], "stop_reason": stop,
                     "input_tokens": inp, "output_tokens": out, "usd": cost,
                     "valid": valid, "error": error, "result": parsed})
        print(f"{doc['id']} [{number}/{len(prepared)}] {time.time()-started:.1f}s "
              f"${cost:.4f} valid={valid} stop={stop}")
        if not valid or stop != "stop":
            print("STOPPING: incomplete or invalid response; no further dossiers sent.")
            break
    payload = {"model": MODEL, "reasoning_effort": args.reasoning_effort,
               "documents_requested": len(docs), "rows": rows,
               "ledger": ledger, "cap_usd": args.max_usd}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
