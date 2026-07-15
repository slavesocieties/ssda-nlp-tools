"""Batched, cache-friendly, single-pass extractor — the concrete cost recipe.

The cost model shows the dominant expense is the static few-shot prefix billed
once PER ENTRY. Two quality-preserving fixes, both implemented here:

  1. BATCH: put N entries in one call so the prefix is paid once per batch, not
     per entry (the biggest lever — ~Nx on the prefix).
  2. FOLD NORMALIZATION: ask for `normalized` AND `data` in the same call, so the
     separate normalization LLM pass disappears (2 passes -> 1).

Plus CACHE-ORDERING: every static token (system prompt, instructions, few-shots)
comes first and byte-identically across calls, so provider prompt-caching hits
the whole prefix; only the batch of entries varies at the tail.

This module builds the request messages and parses the response. It does NOT
call any API (drop the messages into your OpenAI/Anthropic/Gemini client). That
keeps it fully offline-testable: message construction, cache-ordering, response
parsing, and the token saving are all verified without a network call.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .cost import count_tokens
from .fixes import fix_relationships

BATCH_SYSTEM_PROMPT = """
You are a historical sacramental-register information extraction assistant.
You will receive SEVERAL numbered entries at once. For EACH entry, return both a
lightly normalized transcription and structured people/events data.

Return exactly one JSON object of the form:
{"results": [
   {"entry": "<the entry id you were given>",
    "normalized": "<lightly normalized transcription: fix spacing, line-break "
                  "hyphenation and obvious abbreviations; preserve meaning and names>",
    "data": {"people": [...], "events": [...]}}
]}

Rules:
- One results element per input entry, echoing its exact "entry" id. Do not merge
  or drop entries. If an entry is unreadable, return it with empty people/events.
- people/events follow the same schema and conventions as the few-shot examples.
- Output English-normalized values only for "age" and "origin"; never translate names.
- Prefer omission/null over hallucination. Return JSON only, no prose or code fences.
""".strip()


def build_messages(entries: List[dict], examples: List[dict],
                   instructions: List[dict], record_type: str = "baptism",
                   language: str = "Spanish") -> List[Dict[str, str]]:
    """Assemble a cache-ordered messages array for ONE batch of entries.

    Order (static -> dynamic) is deliberate: identical prefix across every batch
    of a volume maximizes prompt-cache hits; only the final user turn changes.
    """
    msgs: List[Dict[str, str]] = [{"role": "system", "content": BATCH_SYSTEM_PROMPT}]
    for ins in instructions:                       # static project instructions
        text = ins["text"] if isinstance(ins, dict) else str(ins)
        msgs.append({"role": "system", "content": text})
    for ex in examples:                            # static few-shot demonstrations
        msgs.append({"role": "user",
                     "content": f"Example {ex.get('language', language)} "
                                f"{ex.get('type', record_type)} transcription:\n"
                                + ex.get("normalized", "")})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"normalized": ex.get("normalized", ""),
                                            "data": ex.get("data", {})},
                                           ensure_ascii=False)})
    # dynamic tail: the batch itself
    payload = {"instruction": f"Process ALL {len(entries)} {language} {record_type} "
                              "entries below. Return the results array only.",
               "entries": [{"entry": str(e.get("entry") or e.get("id") or i),
                            "transcription": e.get("raw") or e.get("normalized") or ""}
                           for i, e in enumerate(entries)]}
    msgs.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    return msgs


def parse_response(text: str, expected_ids: List[str],
                   validate: bool = True) -> Tuple[Dict[str, dict], List[str]]:
    """Split a batch response into {entry_id: {normalized, data}} + missing ids.

    Robust to code fences and to the model returning a bare array. Applies the
    reciprocity fixer to each entry's data when validate=True.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        obj = json.loads(cleaned[s:e + 1]) if s >= 0 and e > s else {"results": []}

    results = obj.get("results", obj) if isinstance(obj, dict) else obj
    if not isinstance(results, list):
        results = []

    out: Dict[str, dict] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        eid = str(r.get("entry", ""))
        data = r.get("data") or {"people": [], "events": []}
        if validate:
            data, _ = fix_relationships(data)
        out[eid] = {"normalized": r.get("normalized", ""), "data": data}

    missing = [eid for eid in expected_ids if eid not in out]
    return out, missing


def plan_batches(entries: List[dict], batch_size: int = 10) -> List[List[dict]]:
    return [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]


def token_report(entries: List[dict], examples: List[dict], instructions: List[dict],
                 batch_size: int = 10, shots: Optional[int] = None) -> Dict[str, Any]:
    """Compare per-entry vs batched INPUT tokens on real data (offline proof)."""
    ex = examples[:shots] if shots is not None else examples
    # per-entry (current style): prefix rebuilt for every entry
    prefix_msgs = build_messages([], ex, instructions)
    prefix_tokens = sum(count_tokens(m["content"]) for m in prefix_msgs)
    per_entry_tail = int(sum(count_tokens(e.get("raw") or e.get("normalized") or "")
                             for e in entries) / max(1, len(entries))) + 30

    n = len(entries)
    per_entry_total_in = n * (prefix_tokens + per_entry_tail)

    batches = plan_batches(entries, batch_size)
    batched_total_in = 0
    for b in batches:
        msgs = build_messages(b, ex, instructions)
        batched_total_in += sum(count_tokens(m["content"]) for m in msgs)

    # with caching, the prefix on calls 2..k is a cache hit; approximate the
    # billable input as prefix once + tails + (k-1)*prefix*0.1 handled in cost.py.
    return {
        "entries": n, "batch_size": batch_size, "batches": len(batches),
        "shots": len(ex), "prefix_tokens": prefix_tokens,
        "per_entry_input_tokens": per_entry_total_in,
        "batched_input_tokens": batched_total_in,
        "input_reduction_x": round(per_entry_total_in / max(1, batched_total_in), 2),
        "separate_normalization_calls_saved": n,   # folded -> 0 extra calls
    }
