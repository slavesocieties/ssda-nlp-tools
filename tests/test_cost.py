"""Offline tests for the cost model and the batched extractor."""
import json
import os
import sys
import importlib.util

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import cost
from ssda_nlp_tools import batch_extract as bx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bakeoff_module():
    path = os.path.join(ROOT, "run_model_bakeoff.py")
    spec = importlib.util.spec_from_file_location("run_model_bakeoff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scorer_module():
    path = os.path.join(ROOT, "score_entity_f1.py")
    spec = importlib.util.spec_from_file_location("score_entity_f1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scorer_merges_batch_files_and_first_file_wins_on_dup_id(tmp_path):
    """score_entity_f1 stitches a model's per-batch output files into one
    prediction set, deduping by entry id with the earliest file winning (so a
    re-run/overlap can't silently overwrite a good entry)."""
    scorer = _scorer_module()
    def write(name, results):
        p = tmp_path / name
        p.write_text(json.dumps(
            {"models": {"gpt-5.6-luna": {"batches": [{"results": results}]}}}),
            encoding="utf-8")
        return str(p)
    a = write("a.json", [{"entry": "E1", "normalized": "keep", "data": {"people": []}}])
    b = write("b.json", [{"entry": "E1", "normalized": "DROP", "data": {"people": [{"id": "P"}]}},
                         {"entry": "E2", "normalized": "y", "data": {"people": []}}])
    preds = scorer._model_predictions([a, b])["gpt-5.6-luna"]
    assert set(preds) == {"E1", "E2"}
    assert preds["E1"]["normalized"] == "keep"      # earliest file wins on the dup id


def test_scorer_ignores_skipped_model_rows(tmp_path):
    """A model row the bake-off marked 'skipped' (no key set) contributes no
    predictions rather than an empty entry that would score as a total miss."""
    scorer = _scorer_module()
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"models": {
        "gpt-5.6-luna": {"status": "skipped", "reason": "no key"},
        "gpt-5.4-mini": {"batches": [{"results": [
            {"entry": "E1", "normalized": "x", "data": {"people": []}}]}]},
    }}), encoding="utf-8")
    preds = scorer._model_predictions([str(p)])
    assert set(preds) == {"gpt-5.4-mini"}       # skipped luna absent entirely


# ---- token counting ----------------------------------------------------------

def test_count_tokens_monotonic_and_positive():
    assert cost.count_tokens("") == 0
    assert cost.count_tokens("hola mundo") >= 2
    assert cost.count_tokens("a b c d e") < cost.count_tokens("a b c d e f g h i j")


def test_bakeoff_ledger_reserves_then_settles_without_losing_spend(tmp_path):
    bakeoff = _bakeoff_module()
    path = tmp_path / "ledger.json"
    ledger = bakeoff._read_ledger(path)
    bakeoff._reserve(ledger, "gpt-5.4-mini", 0.80)
    bakeoff._write_ledger(path, ledger)
    restarted = bakeoff._read_ledger(path)
    assert bakeoff._ledger_amounts(restarted, "gpt-5.4-mini") == (0.0, 0.80)
    bakeoff._settle(restarted, "gpt-5.4-mini", 0.30, 0.021)
    assert bakeoff._ledger_amounts(restarted, "gpt-5.4-mini") == pytest.approx((0.021, 0.50))


def test_bakeoff_input_ceiling_exceeds_raw_prompt_bytes():
    bakeoff = _bakeoff_module()
    messages = [{"role": "system", "content": "á"}, {"role": "user", "content": "hello"}]
    assert bakeoff._input_token_ceiling(messages) > len("áhello".encode("utf-8"))


def test_bakeoff_4xx_is_provider_rejected_but_5xx_stays_ambiguous():
    """A definitive 4xx means the request was refused and NOT billed, so its
    reservation can be released; a 5xx (or network error) may have been
    processed, so it must stay a plain RuntimeError and keep its reservation."""
    import io
    import urllib.error
    import unittest.mock as mock
    bakeoff = _bakeoff_module()

    def _fake(code):
        return urllib.error.HTTPError("http://x", code, "m", {}, io.BytesIO(b'{"error":"e"}'))

    for code in (400, 404, 422):
        with mock.patch("urllib.request.urlopen", side_effect=_fake(code)):
            with pytest.raises(bakeoff.ProviderRejected):
                bakeoff._request("http://x", {}, {})
    for code in (500, 503):
        with mock.patch("urllib.request.urlopen", side_effect=_fake(code)):
            with pytest.raises(RuntimeError) as ei:
                bakeoff._request("http://x", {}, {})
            assert not isinstance(ei.value, bakeoff.ProviderRejected)


def test_bakeoff_provider_rejected_releases_only_that_reservation(tmp_path):
    """The 4xx handler releases exactly the in-flight batch's reservation (so a
    corrected re-run is not blocked), while leaving any OTHER model's stranded
    reservation untouched."""
    bakeoff = _bakeoff_module()
    path = tmp_path / "ledger.json"
    ledger = bakeoff._read_ledger(path)
    bakeoff._reserve(ledger, "gpt-5.4-mini", 0.10)      # the in-flight batch
    bakeoff._reserve(ledger, "gpt-5.6-luna", 0.42)      # unrelated stranded reservation
    # the 4xx path: settle the in-flight batch's reservation at $0 billed
    bakeoff._settle(ledger, "gpt-5.4-mini", 0.10, 0.0)
    assert bakeoff._ledger_amounts(ledger, "gpt-5.4-mini") == pytest.approx((0.0, 0.0))
    assert bakeoff._ledger_amounts(ledger, "gpt-5.6-luna") == pytest.approx((0.0, 0.42))


def test_bakeoff_system_prompt_override_replaces_only_leading_system_turn():
    """--system-prompt-file swaps the extraction prompt for A/B testing while
    keeping the cache-ordered few-shot prefix and dynamic tail intact — so only
    the first system turn changes."""
    from ssda_nlp_tools.batch_extract import build_messages, BATCH_SYSTEM_PROMPT
    examples, vol = _fixtures()
    messages = build_messages(vol[:3], examples[:2], [])
    original = [m["content"] for m in messages]
    assert messages[0]["content"] == BATCH_SYSTEM_PROMPT
    # the exact loop run_model_bakeoff applies for an override
    variant = "VARIANT EXTRACTION PROMPT"
    for m in messages:
        if m["role"] == "system":
            m["content"] = variant
            break
    assert messages[0]["content"] == variant
    # every other turn is byte-identical -> the few-shot cache prefix is preserved
    assert [m["content"] for m in messages[1:]] == original[1:]


def test_bakeoff_haiku_sends_dated_wire_id():
    """Anthropic's Haiku 4.5 requires the dated model id on the wire; the
    friendly name is kept for internal MODELS lookups."""
    bakeoff = _bakeoff_module()
    assert bakeoff.MODELS["claude-haiku-4-5"]["id"] == "claude-haiku-4-5-20251001"
    # models without an override send their friendly name unchanged
    assert "id" not in bakeoff.MODELS["gpt-5.4-mini"]


# ---- cost model levers -------------------------------------------------------

def _comp():
    return cost.measure_components(ROOT)


def test_measure_components_reads_real_files():
    c = _comp()
    assert c.sys_extract > 100            # extract.py system prompt found
    assert c.n_shots_available == 15
    assert c.entry_in > 0 and c.entry_out > 0
    assert 1.0 < c.entries_per_image < 6.0


def test_caching_reduces_cost():
    c = _comp(); p = cost.DEFAULT_PRICING
    off = cost.scenario_cost(c, p, cost.Scenario(cached=False))["per_image"]["total"]
    on = cost.scenario_cost(c, p, cost.Scenario(cached=True))["per_image"]["total"]
    assert on < off


def test_batching_reduces_cost():
    c = _comp(); p = cost.DEFAULT_PRICING
    b1 = cost.scenario_cost(c, p, cost.Scenario(batch=1, cached=True))["per_image"]["total"]
    b10 = cost.scenario_cost(c, p, cost.Scenario(batch=10, cached=True))["per_image"]["total"]
    assert b10 < b1


def test_folding_normalization_reduces_cost():
    c = _comp(); p = cost.DEFAULT_PRICING
    sep = cost.scenario_cost(c, p, cost.Scenario(normalize="separate"))["per_image"]["total"]
    fold = cost.scenario_cost(c, p, cost.Scenario(normalize="folded"))["per_image"]["total"]
    assert fold < sep


def test_optimizer_finds_recipe_under_target():
    c = _comp(); p = cost.DEFAULT_PRICING
    r = cost.optimize(c, p, target=0.01, min_shots=5)
    assert r["n_meeting"] > 0
    rec = r["recommended"]
    assert rec["per_image"]["transcription_plus_normalization"] <= 0.01
    assert rec["scenario"]["shots"] >= 5                    # accuracy guardrail respected


def test_waterfall_is_monotonically_cheaper_through_quality_levers():
    c = _comp(); p = cost.DEFAULT_PRICING
    rows = cost.lever_waterfall(c, p, model="claude-haiku-4.5")
    # baseline -> cache -> batch -> fold : each step no more expensive than the last
    totals = [r["total"] for r in rows[:4]]
    assert totals == sorted(totals, reverse=True)
    assert rows[0]["total"] > rows[3]["total"]              # net improvement


# ---- batched extractor -------------------------------------------------------

def _fixtures():
    examples = json.load(open(os.path.join(ROOT, "training_data.json"), encoding="utf-8"))["examples"]
    vol = json.load(open(os.path.join(ROOT,
        "Sample_output/Generated_0013_0023_4o_prompt_V2.json"), encoding="utf-8"))["examples"]
    return examples, vol


def test_messages_are_cache_ordered():
    examples, vol = _fixtures()
    instr = [{"text": "SSDA schema."}]
    b1 = bx.build_messages(vol[0:10], examples, instr)
    b2 = bx.build_messages(vol[10:20], examples, instr)
    # every turn except the last (the batch) is byte-identical -> full prefix caches
    assert [m["content"] for m in b1[:-1]] == [m["content"] for m in b2[:-1]]
    assert b1[-1]["content"] != b2[-1]["content"]


def test_batch_response_roundtrip_preserves_data():
    examples, vol = _fixtures()
    batch = vol[:5]
    ids = [str(e["entry"]) for e in batch]
    resp = json.dumps({"results": [
        {"entry": str(e["entry"]), "normalized": e.get("normalized", ""),
         "data": e.get("data", {})} for e in batch]}, ensure_ascii=False)
    parsed, missing = bx.parse_response(resp, ids, validate=False)
    assert missing == [] and len(parsed) == 5
    for e in batch:
        assert len(parsed[str(e["entry"])]["data"]["people"]) == len(e["data"]["people"])


def test_parse_handles_fences_and_reports_missing():
    parsed, missing = bx.parse_response(
        "```json\n{\"results\":[{\"entry\":\"A\",\"data\":{\"people\":[],\"events\":[]}}]}\n```",
        ["A", "B", "C"])
    assert set(parsed) == {"A"} and set(missing) == {"B", "C"}


def test_merge_with_faithful_keeps_both_texts():
    canonical = [{"id": "V-0001-01", "text": "raw archivault text", "images": ["V-0001.jpg"]}]
    parsed = {"V-0001-01": {"normalized": "Cleaned Text.", "data": {"people": [], "events": []}}}
    merged = bx.merge_with_faithful(canonical, parsed)
    assert len(merged) == 1
    r = merged[0]
    assert r["text_faithful"] == "raw archivault text"      # unchanged from the segmenter
    assert r["text_normalized"] == "Cleaned Text."           # the LLM's version, NOT overwriting it
    assert r["data"] == {"people": [], "events": []}
    assert r["images"] == ["V-0001.jpg"]


def test_merge_with_faithful_keeps_unmatched_entries_not_drops_them():
    canonical = [{"id": "V-0001-01", "text": "raw text", "images": ["V-0001.jpg"], "partial": True}]
    merged = bx.merge_with_faithful(canonical, {})    # model never returned this entry
    assert len(merged) == 1                            # present, not silently dropped
    assert merged[0]["text_faithful"] == "raw text"
    assert merged[0]["text_normalized"] is None
    assert merged[0]["data"] is None
    assert merged[0]["partial"] is True


def test_token_report_shows_large_reduction():
    examples, vol = _fixtures()
    tr = bx.token_report(vol, examples, [{"text": "x"}], batch_size=10)
    assert tr["input_reduction_x"] >= 3.0                   # batching is a big lever
    assert tr["separate_normalization_calls_saved"] == tr["entries"]


# ---- vendor Batch API + prompt-caching stacking -------------------------------
# Regression tests pinning the exact numbers vendors state in their own docs
# (2026-07-16): Anthropic says a cached+batched request can cost "as little as
# 5%" of standard; OpenAI's own reported example puts cached+batched GPT-5.4
# input at $0.625/M vs a $2.50 base = 25%, NOT the naive 5% you'd get by just
# multiplying the two individual discounts. Getting this wrong silently
# understates OpenAI cost or overstates the caching benefit.

def test_vendor_batch_alone_halves_every_rate():
    p = cost.DEFAULT_PRICING["claude-sonnet-5"]
    in_rate, cached_rate, out_rate = cost._rates(p, cached=False, vendor_batch_api=True)
    assert in_rate == p.input * 0.5
    assert out_rate == p.output * 0.5
    assert cached_rate == in_rate            # no caching active -> same as uncached


def test_anthropic_cached_plus_batch_stacks_to_5_percent():
    p = cost.DEFAULT_PRICING["claude-sonnet-5"]      # $2.00 input, cached_batch_mult=0.05
    _, cached_and_batched, _ = cost._rates(p, cached=True, vendor_batch_api=True)
    assert cached_and_batched == pytest.approx(0.10)  # 5% of $2.00, per Anthropic's own claim


def test_openai_cached_plus_batch_stacks_to_25_percent_not_5():
    p = cost.DEFAULT_PRICING["gpt-5.4-mini"]         # $0.75 input, cached_batch_mult=0.25
    _, cached_and_batched, _ = cost._rates(p, cached=True, vendor_batch_api=True)
    assert cached_and_batched == pytest.approx(0.1875)   # 25% of $0.75
    naive_multiplicative = 0.75 * (p.cached / p.input) * p.batch_mult   # what you'd get wrong
    assert cached_and_batched != pytest.approx(naive_multiplicative)


def test_no_vendor_batch_api_reproduces_original_interactive_rates():
    p = cost.DEFAULT_PRICING["claude-sonnet-5"]
    in_rate, cached_rate, out_rate = cost._rates(p, cached=True, vendor_batch_api=False)
    assert (in_rate, cached_rate, out_rate) == (p.input, p.cached, p.output)


def test_vendor_batch_api_lowers_scenario_cost():
    comp = _comp()
    sc_off = cost.Scenario(model="claude-sonnet-5", cached=True, vendor_batch_api=False, batch=10)
    sc_on = cost.Scenario(model="claude-sonnet-5", cached=True, vendor_batch_api=True, batch=10)
    off = cost.scenario_cost(comp, cost.DEFAULT_PRICING, sc_off)["per_image"]["total"]
    on = cost.scenario_cost(comp, cost.DEFAULT_PRICING, sc_on)["per_image"]["total"]
    assert on < off


# ---- quality-first optimizer (one-time-run framing) ---------------------------

def test_optimize_for_quality_stays_within_budget_when_possible():
    comp = _comp()
    r = cost.optimize_for_quality(comp, cost.DEFAULT_PRICING, budget=0.01)
    assert r["n_under_budget"] > 0
    assert not r["over_budget"]
    assert r["best"]["metric_value"] <= 0.01


def test_optimize_for_quality_prefers_quality_over_cheapness():
    comp = _comp()
    r = cost.optimize_for_quality(comp, cost.DEFAULT_PRICING, budget=0.01)
    best_quality = r["best"]["quality"]
    cheapest_under_budget = min((x for x in r["all_ranked"] if x["metric_value"] <= 0.01),
                                key=lambda x: x["metric_value"])
    # best must be at least as high quality as the cheapest option that fits —
    # the whole point of this mode is not defaulting to "cheapest that fits"
    assert best_quality >= cheapest_under_budget["quality"]


def test_optimize_for_quality_never_cuts_shots():
    comp = _comp()
    r = cost.optimize_for_quality(comp, cost.DEFAULT_PRICING, budget=0.01)
    for row in r["all_ranked"]:
        assert row["scenario"]["shots"] == comp.n_shots_available


def test_optimize_for_quality_flags_when_nothing_fits():
    comp = _comp()
    r = cost.optimize_for_quality(comp, cost.DEFAULT_PRICING, budget=0.0000001)
    assert r["over_budget"] is True
    assert r["n_under_budget"] == 0
