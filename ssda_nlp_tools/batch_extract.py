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
You are a historical sacramental-register normalization and extraction assistant.
You will receive SEVERAL numbered entries at once. For EACH entry, return both a
normalized transcription and structured people/events data.

Return exactly one JSON object of the form:
{"results": [
   {"entry": "<the entry id you were given>",
    "normalized": "<normalized transcription — see rules>",
    "data": {"people": [...], "events": [...]}}
]}

Normalization rules (match the project's hand-corrected examples):
- REMOVE interleaved margin names: register pages have a name column in the left
  margin that the transcriber merges into the body lines ("...Bernardo Cae
  da Concepcion tano de Freitas..."). Remove those margin words and re-join the
  word they interrupt ("Cae"+"tano" -> "Caetano").
- Heal words broken across line wraps (with or without "-"/"=" marks).
- Expand scribal abbreviations to full words ("R.do P.e fr." -> "Reverendo Padre
  fray", "leg.mo" -> "legitimo", "fuer.n sus Padr.os" -> "fueron sus Padrinos").
- Modernize archaic spelling to standard modern orthography of the entry's own
  language, with accents ("dezasete" -> "dezessete", "annos" -> "anos",
  "Xuarez" -> "Juarez").
- LANGUAGE — CRITICAL: the normalized text MUST stay in the SAME language as the
  source. A PORTUGUESE record stays Portuguese ("Aos vinte dias do mês de Maio",
  "sepultou-se", "anos", "freguesia") — do NOT render it in Spanish ("A los
  veinte días del mes de mayo", "se sepultó", "años"). A Spanish record stays
  Spanish. Never translate one Iberian language into the other; the few-shot
  examples happen to be Spanish, but they do not change the source's language.
- Repair only OBVIOUS transcription dropouts recoverable from formulaic context
  ("de mil ochocien noventa" -> "de mil ochocientos noventa"). Never invent
  names, dates, or facts.
- If an entry is marked as possibly truncated, normalize what is present; do not
  guess a continuation.

Extraction rules:
- One results element per input entry, echoing its exact "entry" id. Do not merge
  or drop entries. If an entry is unreadable, return it with empty people/events.
- Include a field ONLY when the record actually supports it. Omit it otherwise.
  Never guess a value to fill a slot — omission is always better than invention.
- Never translate personal names.
- Return JSON only, no prose or code fences.
""".strip()


def _schema_block(v=None) -> str:
    """The field spec, rendered from vocab.json so the prompt cannot drift from
    the controlled vocabularies.

    Semantics follow slavesocieties/openai `training_data_documentation.txt`.
    Only genuinely closed sets are enumerated in full: listing every ethnicity or
    occupation would push the model to force-fit a value onto a record that does
    not state one, and Daniel describes those lists as representative rather than
    comprehensive. For the open-ended fields we give the DISTINCTION plus a few
    examples, which is what the model actually gets wrong.
    """
    from . import vocab as _v
    v = v or _v.load_vocab()

    def _lst(field, lang="English", n=None):
        vals = v.values(field, lang)
        return ", ".join(vals[:n] if n else vals)

    def _spread(field, lang="English", head=8, tail=4):
        """Sample across a long list. The ethnicity vocabulary is grouped —
        African ethnonyms first, European and Indigenous ones last — so a plain
        head slice would imply the field only covers the former."""
        vals = v.values(field, lang)
        if len(vals) <= head + tail:
            return ", ".join(vals)
        return ", ".join(vals[:head] + ["..."] + vals[-tail:])

    return f"""
PERSON fields — include only those the record supports:
- id: "P01", "P02", ... in order of first appearance in the entry.
- name: the person's name as it appears in the NORMALIZED text.
- titles: honorifics, kept in the record's OWN language, as written
  (e.g. {_lst('titles', 'Spanish', 6)}).
- rank: military rank only, given as its ENGLISH equivalent
  (one of: {_lst('rank')}).
- occupation: what the person did for a living, GENERALIZED and in ENGLISH —
  every clergyman is "Cleric" regardless of church office
  (e.g. {_lst('occupation', 'English', 10)}).
- origin: a place — birthplace or vaguer origin. A continent, region, or town.
- ethnicity: the ETHNOLINGUISTIC descriptor only — African, European and
  Indigenous designations all belong here
  (e.g. {_spread('ethnicity')}).
- phenotype: the PHYSICAL/colour descriptor, kept as written in the record
  (e.g. {_lst('phenotype', 'Spanish', 8)}).
  ethnicity and phenotype are DIFFERENT fields and must never be merged: an
  ethnicity names a people or place of derivation, a phenotype describes
  appearance. "criollo/criolla" is a PHENOTYPE. Neither belongs in origin,
  which is strictly a place name.
- age: exactly one of {_lst('age')} — a category, never a number and never the
  Spanish or Portuguese word. "párvulo"/"párvula" and any child under two are
  "infant". Infer from an explicit or referenced birth date when present.
- age_stated: when the record gives a literal age, ALSO record it here exactly
  as written ("64 años", "de un mes", "tres años"). `age` stays the bucket;
  this preserves the evidence. Omit when the record states no age.
- legitimate: boolean. true = born in wedlock, false = not. Often implicit:
  "hijo natural" and "padres no conocidos" both mean false.
- free: boolean. true = free at the time of the record, false = enslaved.
  "libre"/"liberto"/"horro" are true; "esclavo"/"esclava" false. If the record
  does not say, OMIT the field — do not default it either way.
- relationships: [{{"related_person": "<id>", "relationship_type": "<type>"}}].
  relationship_type MUST be one of exactly: {_lst('relationship_type')}.
  Use the ENGLISH term even though the record is Spanish or Portuguese
  (padrino/madrinha -> godparent, esclavo -> slave, amo/senhor -> enslaver,
  hijo/filha -> child). Record the relationship from BOTH sides: if P03 is
  "hija de P05", then P05 is P03's "parent" and P03 is P05's "child".
  DIRECTION MATTERS. Most of these are asymmetric pairs, and the two sides take
  DIFFERENT terms: parent/child, grandparent/grandchild, enslaver/slave,
  godparent/godchild, patron/client, custodian/ward, executor/testator,
  heir/benefactor, caregiver/dependent. Never label both sides with the same
  term for these — "patron" on both people loses which one held the patronage.
  Only "spouse", "former spouse" and "witness" are symmetric and take the same
  term on both sides.

EVENT fields:
- type: "baptism", "marriage", "burial", or "birth". A baptismal entry
  frequently also carries a separate birth event.
- principals: [person id]. One for baptism/burial/birth; TWO for a marriage.
- date: "YYYY-MM-DD". If the record gives only part of a date, give what it
  supports ("1798-06" or "1798") rather than inventing the rest.
- witnesses: [person id] for anyone named as a witness (testigo/testemunha).
  Most common in marriages. Omit when the record names none.
""".rstrip()


BATCH_SYSTEM_PROMPT = BATCH_SYSTEM_PROMPT + "\n" + _schema_block()


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


def merge_with_faithful(canonical_entries: List[dict],
                        parsed: Dict[str, dict]) -> List[dict]:
    """Join the segmenter's FAITHFUL text (exactly what Archivault produced —
    see segment.to_canonical) with the LLM's NORMALIZED text + extracted data,
    by entry id. Confirmed decision (2026-07-16): keep BOTH, not just the
    normalized version — faithful is the auditable record of truth (free,
    reproducible, no model-invented text), normalized is what a researcher
    actually wants to read. Neither replaces the other in the output.

    canonical_entries: the output of segment.to_canonical() — [{id, text,
        images, partial?}, ...] where "text" is the faithful segmented text.
    parsed: the output of parse_response() — {id: {normalized, data}}.

    Entries with no matching parsed result (not yet sent, or the model dropped
    them) still appear, with normalized/data left null — never silently
    dropped, consistent with how partial records are handled elsewhere here.
    """
    merged = []
    for e in canonical_entries:
        eid = e["id"]
        p = parsed.get(eid)
        row = {"id": eid, "images": e.get("images", []),
              "text_faithful": e["text"],
              "text_normalized": p["normalized"] if p else None,
              "data": p["data"] if p else None}
        if e.get("partial"):
            row["partial"] = True
        merged.append(row)
    return merged


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
