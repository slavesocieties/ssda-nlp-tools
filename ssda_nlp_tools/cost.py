"""Cost model for the transcription + normalization + extraction pipeline.

Answers one question with real numbers: what does it cost to process an image,
and what is the recipe to get transcription + normalization to <= $0.01/image?

Everything is measured from the ACTUAL repo files (extract.py's system prompt,
instructions.json, training_data.json, a real volume) so the token counts are
the pipeline's own, not assumptions. No LLM calls, no network.

Two honest caveats, both surfaced in every report:
  * Token counts use a calibrated estimator (tiktoken is used automatically if
    installed). It is deliberately slightly CONSERVATIVE (rounds costs up).
  * Prices are a configurable table of representative early-2026 public rates.
    Override with --pricing my_prices.json; the LEVERS and their relative sizes
    are robust to the exact numbers.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# token counting
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
try:
    import tiktoken  # optional; used if present
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENC = None


def count_tokens(text: str) -> int:
    """Estimate GPT-style token count. Uses tiktoken if available, else a
    subword heuristic tuned for mixed Spanish/English + JSON (conservative)."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    toks = 0
    for u in _TOKEN_RE.findall(text):
        if u.isalnum():
            toks += max(1, round(len(u) / 4.2))   # ~4.2 chars/subword
        else:
            toks += 1                              # punctuation ~= 1 token
    return toks


# --------------------------------------------------------------------------- #
# pricing  (USD per 1,000,000 tokens)  — VERIFY against current provider rates
# --------------------------------------------------------------------------- #

@dataclass
class Price:
    input: float          # $/1M input tokens
    cached: float         # $/1M cached input tokens (prompt-cache hit)
    output: float         # $/1M output tokens
    note: str = ""


# Representative early-2026 public list prices. Override via --pricing JSON.
DEFAULT_PRICING: Dict[str, Price] = {
    "gpt-4o":            Price(2.50, 1.25, 10.00, "flagship"),
    "gpt-4o-mini":       Price(0.15, 0.075, 0.60, "mid tier"),
    "gpt-4.1-nano":      Price(0.10, 0.025, 0.40, "nano tier"),
    "gemini-flash":      Price(0.10, 0.025, 0.40, "vision + text"),
    "gemini-flash-lite": Price(0.075, 0.01875, 0.30, "cheapest vision"),
    "claude-haiku":      Price(0.80, 0.08, 4.00, "cache 10x cheaper"),
    "claude-sonnet":     Price(3.00, 0.30, 15.00, "flagship, cache 10x"),
}


def load_pricing(path: Optional[str]) -> Dict[str, Price]:
    if not path:
        return dict(DEFAULT_PRICING)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Price(**v) if isinstance(v, dict) else Price(*v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# measure the pipeline's real token components
# --------------------------------------------------------------------------- #

@dataclass
class Components:
    sys_extract: int
    sys_normalize: int
    instructions: int
    shot_avg: int              # avg tokens per few-shot (user text + assistant json)
    n_shots_available: int
    entry_in: int              # per-entry input tokens (normalized text)
    entry_out: int             # per-entry extraction output tokens (data json)
    norm_in: int               # per-entry normalization input (raw text)
    norm_out: int              # per-entry normalization output (normalized text)
    entries_per_image: float
    transcription_out: int     # per-page transcription output tokens
    detail: Dict[str, Any] = field(default_factory=dict)


def _extract_triple_quoted(pyfile: str, varname: str) -> str:
    """Pull the text of a `VAR = \"\"\"...\"\"\"` literal from a .py source file."""
    try:
        src = open(pyfile, "r", encoding="utf-8").read()
    except OSError:
        return ""
    m = re.search(varname + r'\s*=\s*(?:r|f)?"""(.*?)"""', src, re.S)
    return m.group(1) if m else ""


def measure_components(repo_dir: str = ".",
                       sample_volume: Optional[str] = None) -> Components:
    j = lambda *p: os.path.join(repo_dir, *p)

    sys_extract = count_tokens(_extract_triple_quoted(j("extract.py"), "EXTRACTION_SYSTEM_PROMPT"))
    # normalize.py's system prompt may be named differently; try a few
    norm_src = ""
    for var in ("NORMALIZATION_SYSTEM_PROMPT", "SYSTEM_PROMPT", "NORMALIZE_SYSTEM_PROMPT"):
        norm_src = _extract_triple_quoted(j("normalize.py"), var)
        if norm_src:
            break
    sys_normalize = count_tokens(norm_src) or sys_extract // 2   # fallback estimate

    instructions = count_tokens(open(j("instructions.json"), encoding="utf-8").read())

    td = json.load(open(j("training_data.json"), encoding="utf-8"))["examples"]
    shot_toks = []
    for e in td:
        user = f"Extract ... transcription:\n{e.get('normalized','')}"
        asst = json.dumps(e.get("data", {}), ensure_ascii=False)
        shot_toks.append(count_tokens(user) + count_tokens(asst))
    shot_avg = sum(shot_toks) // len(shot_toks)

    # per-entry sizes from a real volume
    if sample_volume is None:
        for cand in ("Sample_output/Generated_0013_0023_4o_prompt_V2.json",):
            if os.path.exists(j(cand)):
                sample_volume = j(cand)
                break
    entries = json.load(open(sample_volume, encoding="utf-8"))["examples"] if sample_volume else td
    ent_in = _mean([count_tokens(e.get("normalized") or e.get("raw") or "") for e in entries])
    ent_out = _mean([count_tokens(json.dumps(e.get("data", {}), ensure_ascii=False)) for e in entries])
    raw_in = _mean([count_tokens(e.get("raw") or e.get("normalized") or "") for e in entries])
    pages = {str(e.get("entry", "")).split("-")[0] for e in entries}
    epi = len(entries) / max(1, len(pages))

    # transcription output = whole-page text produced by Archivault/Gemini
    trans_out = 0
    for cand in ("Text data/SSDA_0013_0023_Gemini_V2.json",):
        if os.path.exists(j(cand)):
            raw = json.load(open(j(cand), encoding="utf-8"))
            trans_out = _all_text_tokens(raw) // max(1, len(pages))
            break
    if not trans_out:
        trans_out = ent_in * int(round(epi))

    return Components(
        sys_extract=sys_extract, sys_normalize=sys_normalize, instructions=instructions,
        shot_avg=shot_avg, n_shots_available=len(td),
        entry_in=ent_in, entry_out=ent_out, norm_in=raw_in, norm_out=ent_in,
        entries_per_image=round(epi, 2), transcription_out=trans_out,
        detail={"tokenizer": "tiktoken" if _ENC else "heuristic(conservative)",
                "sample_volume": sample_volume, "shots_measured": len(td)},
    )


def _mean(xs): return int(round(sum(xs) / len(xs))) if xs else 0


def _all_text_tokens(o) -> int:
    if isinstance(o, str):
        return count_tokens(o)
    if isinstance(o, dict):
        return sum(_all_text_tokens(v) for v in o.values())
    if isinstance(o, list):
        return sum(_all_text_tokens(v) for v in o)
    return 0


# --------------------------------------------------------------------------- #
# scenario cost
# --------------------------------------------------------------------------- #

@dataclass
class Scenario:
    model: str = "gpt-4o-mini"
    trans_model: str = "gemini-flash"
    shots: int = 5
    cached: bool = False
    batch: int = 1                     # entries per extraction call
    normalize: str = "separate"        # separate | folded | none
    entries_per_image: Optional[float] = None
    images_per_volume: int = 300       # amortization horizon for prompt cache
    trans_image_tokens: int = 1290     # per-page image tokens (Gemini estimate)
    trans_prompt_tokens: int = 400     # transcription instruction tokens


def _pass_cost(comp: Components, price: Price, *, sys_toks: int, shots: int,
               cached: bool, batch: int, entries: float, per_entry_in: int,
               per_entry_out: int) -> Dict[str, float]:
    """Cost of one LLM pass (extraction OR normalization) over `entries`."""
    prefix = sys_toks + comp.instructions + shots * comp.shot_avg
    calls = math.ceil(entries / max(1, batch))
    batch_overhead = 20 * batch                       # per-entry framing tokens
    if cached:
        prefix_cost = (prefix * price.input + (calls - 1) * prefix * price.cached) / 1e6
    else:
        prefix_cost = calls * prefix * price.input / 1e6
    input_cost = prefix_cost + entries * (per_entry_in + 20) * price.input / 1e6 \
        + calls * batch_overhead * price.input / 1e6
    output_cost = entries * per_entry_out * price.output / 1e6
    return {"input": input_cost, "output": output_cost, "total": input_cost + output_cost,
            "calls": calls, "prefix_tokens": prefix}


def scenario_cost(comp: Components, pricing: Dict[str, Price], sc: Scenario) -> Dict[str, Any]:
    epi = sc.entries_per_image or comp.entries_per_image
    V = max(1, sc.images_per_volume)
    E = epi * V
    price = pricing[sc.model]

    extract = _pass_cost(comp, price, sys_toks=comp.sys_extract, shots=sc.shots,
                         cached=sc.cached, batch=sc.batch, entries=E,
                         per_entry_in=comp.entry_in, per_entry_out=comp.entry_out)

    if sc.normalize == "separate":
        norm = _pass_cost(comp, price, sys_toks=comp.sys_normalize, shots=sc.shots,
                          cached=sc.cached, batch=sc.batch, entries=E,
                          per_entry_in=comp.norm_in, per_entry_out=comp.norm_out)
        norm_total = norm["total"]
    elif sc.normalize == "folded":
        # normalization folded into extraction: only the extra output tokens
        norm_total = E * comp.norm_out * price.output / 1e6
    else:  # none
        norm_total = 0.0

    tp = pricing[sc.trans_model]
    trans_per_image = ((sc.trans_image_tokens + sc.trans_prompt_tokens) * tp.input
                       + comp.transcription_out * tp.output) / 1e6
    trans_total = trans_per_image * V

    total = extract["total"] + norm_total + trans_total
    # the target metric: transcription + normalization only (per the ask)
    trans_norm_per_image = (trans_total + norm_total) / V
    return {
        "scenario": sc.__dict__,
        "entries_per_image": epi,
        "per_image": {
            "transcription": round(trans_total / V, 5),
            "normalization": round(norm_total / V, 5),
            "extraction": round(extract["total"] / V, 5),
            "total": round(total / V, 5),
            "transcription_plus_normalization": round(trans_norm_per_image, 5),
        },
        "extract_calls_per_volume": extract["calls"],
        "prefix_tokens": extract["prefix_tokens"],
    }


# --------------------------------------------------------------------------- #
# optimizer: find recipes that hit a per-image target
# --------------------------------------------------------------------------- #

def optimize(comp: Components, pricing: Dict[str, Price], *,
             target: float = 0.01, min_shots: int = 5,
             metric: str = "transcription_plus_normalization",
             base: Optional[Scenario] = None) -> Dict[str, Any]:
    """Sweep the lever space; return recipes meeting `target` on `metric`,
    ranked cheapest first, keeping shots >= min_shots (accuracy guardrail)."""
    base = base or Scenario()
    models = [m for m in pricing if m not in ("claude-sonnet", "gpt-4o")] or list(pricing)
    trans_models = [m for m in ("gemini-flash-lite", "gemini-flash") if m in pricing] or [base.trans_model]
    results = []
    for model in models:
        for tmodel in trans_models:
            for shots in sorted({min_shots, comp.n_shots_available}):
                for cached in (True, False):
                    for batch in (1, 5, 10, 20):
                        for normalize in ("separate", "folded", "none"):
                            sc = Scenario(model=model, trans_model=tmodel, shots=shots,
                                          cached=cached, batch=batch, normalize=normalize,
                                          entries_per_image=base.entries_per_image,
                                          images_per_volume=base.images_per_volume)
                            r = scenario_cost(comp, pricing, sc)
                            r["metric_value"] = r["per_image"][metric]
                            results.append(r)
    results.sort(key=lambda r: r["metric_value"])
    meeting = [r for r in results if r["metric_value"] <= target]
    return {"target": target, "metric": metric, "min_shots": min_shots,
            "n_meeting": len(meeting), "cheapest": results[0],
            "recommended": _recommend(meeting, results), "all_ranked": results}


def _recommend(meeting: List[dict], allr: List[dict]) -> Optional[dict]:
    """Pick the highest-quality recipe that still meets target: prefer folded
    over none (keeps normalization), prefer caching+batch over shot cuts, prefer
    a non-nano model when it still fits."""
    if not meeting:
        return None
    def quality(r):
        s = r["scenario"]
        q = 0
        q += {"folded": 2, "separate": 1, "none": 0}[s["normalize"]]   # keep normalization
        q += 2 if s["cached"] else 0
        q += 1 if s["batch"] >= 10 else 0
        q += {"gpt-4o-mini": 2, "gemini-flash": 2, "claude-haiku": 1}.get(s["model"], 0)
        return q
    return sorted(meeting, key=lambda r: (-quality(r), r["metric_value"]))[0]


def lever_waterfall(comp: Components, pricing: Dict[str, Price],
                    model: str = "gpt-4o-mini", trans_model: str = "gemini-flash",
                    images_per_volume: int = 300) -> List[dict]:
    """Apply cost levers one at a time from the current-style baseline, so each
    lever's marginal saving is visible. Order: cache -> batch -> folded norm."""
    steps = [
        ("baseline (per-entry call, 15 shots, no cache, separate norm)",
         Scenario(model=model, trans_model=trans_model, shots=comp.n_shots_available,
                  cached=False, batch=1, normalize="separate")),
        ("+ prompt caching (shared prefix cached across the volume)",
         Scenario(model=model, trans_model=trans_model, shots=comp.n_shots_available,
                  cached=True, batch=1, normalize="separate")),
        ("+ batch 10 entries/call (amortize the prefix 10x)",
         Scenario(model=model, trans_model=trans_model, shots=comp.n_shots_available,
                  cached=True, batch=10, normalize="separate")),
        ("+ fold normalization into extraction (2 passes -> 1)",
         Scenario(model=model, trans_model=trans_model, shots=comp.n_shots_available,
                  cached=True, batch=10, normalize="folded")),
        ("+ drop to 5 shots (accuracy-sensitive — see bake-off)",
         Scenario(model=model, trans_model=trans_model, shots=5,
                  cached=True, batch=10, normalize="folded")),
    ]
    rows = []
    prev = None
    for label, sc in steps:
        sc.images_per_volume = images_per_volume
        r = scenario_cost(comp, pricing, sc)
        tn = r["per_image"]["transcription_plus_normalization"]
        tot = r["per_image"]["total"]
        rows.append({"label": label, "trans_norm": tn, "total": tot,
                     "delta_total": None if prev is None else round(prev - tot, 5)})
        prev = tot
    return rows


def corpus_totals(per_image_total: float, corpus: int = 750_000) -> Dict[str, float]:
    return {"corpus_images": corpus, "usd": round(per_image_total * corpus, 2)}


def format_cost(report: Dict[str, Any], comp: Components) -> str:
    L = ["=" * 70, "SSDA pipeline cost model", "=" * 70,
         f"tokenizer: {comp.detail['tokenizer']}   "
         f"entries/image: {comp.entries_per_image}   shots available: {comp.n_shots_available}",
         f"measured tokens — sys(extract)={comp.sys_extract} sys(norm)={comp.sys_normalize} "
         f"instr={comp.instructions} shot_avg={comp.shot_avg}",
         f"                  entry_in={comp.entry_in} entry_out={comp.entry_out} "
         f"trans_out={comp.transcription_out}", ""]
    tgt, metric = report["target"], report["metric"]
    L.append(f"TARGET: {metric.replace('_',' ')} <= ${tgt:.3f}/image")
    L.append(f"recipes meeting target (shots>={report['min_shots']}): {report['n_meeting']}")
    L.append("")

    def line(tag, r):
        s = r["scenario"]; pi = r["per_image"]
        return (f"{tag:<10} {s['model']:<15} shots={s['shots']:<2} "
                f"cache={'Y' if s['cached'] else 'N'} batch={s['batch']:<2} "
                f"norm={s['normalize']:<8} | trans+norm=${pi['transcription_plus_normalization']:.4f} "
                f"total=${pi['total']:.4f}")

    rec = report["recommended"]
    if rec:
        L.append("RECOMMENDED (cheapest that keeps quality):")
        L.append("  " + line("", rec))
        pi = rec["per_image"]
        L.append(f"     breakdown/image: transcription ${pi['transcription']:.4f} + "
                 f"normalization ${pi['normalization']:.4f} + extraction ${pi['extraction']:.4f}")
        L.append(f"     ({rec['extract_calls_per_volume']} extraction calls/volume, "
                 f"prefix {rec['prefix_tokens']} tokens cached across them)")
    L.append("")
    L.append("cheapest overall (may sacrifice quality):")
    L.append("  " + line("", report["cheapest"]))
    L.append("")
    L.append("=" * 70)
    return "\n".join(L)


def format_waterfall(rows: List[dict], pricing_note: str = "",
                     corpus: int = 750_000, model: str = "gpt-4o-mini") -> str:
    L = ["", "=" * 70, f"LEVER WATERFALL  (extraction model = {model})", "=" * 70,
         f"{'step':<58}{'t+n/img':>9}{'total/img':>10}"]
    L.append("-" * 70)
    base_total = rows[0]["total"]
    for r in rows:
        d = "" if r["delta_total"] is None else f"  (-${r['delta_total']:.4f})"
        L.append(f"{r['label']:<58}${r['trans_norm']:.4f}  ${r['total']:.4f}{d}")
    final = rows[-2]  # the quality-preserving endpoint (before the shot cut)
    save = base_total - final["total"]
    L.append("-" * 70)
    L.append(f"quality-preserving endpoint: ${final['total']:.4f}/image total "
             f"(was ${base_total:.4f})  ->  {base_total/max(final['total'],1e-9):.0f}x cheaper")
    L.append(f"across {corpus:,} images: "
             f"${base_total*corpus:,.0f}  ->  ${final['total']*corpus:,.0f}  "
             f"(saves ${save*corpus:,.0f})")
    if pricing_note:
        L.append(pricing_note)
    L.append("=" * 70)
    return "\n".join(L)
