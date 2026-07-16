"""Offline tests for the cost model and the batched extractor."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssda_nlp_tools import cost
from ssda_nlp_tools import batch_extract as bx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- token counting ----------------------------------------------------------

def test_count_tokens_monotonic_and_positive():
    assert cost.count_tokens("") == 0
    assert cost.count_tokens("hola mundo") >= 2
    assert cost.count_tokens("a b c d e") < cost.count_tokens("a b c d e f g h i j")


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


def test_token_report_shows_large_reduction():
    examples, vol = _fixtures()
    tr = bx.token_report(vol, examples, [{"text": "x"}], batch_size=10)
    assert tr["input_reduction_x"] >= 3.0                   # batching is a big lever
    assert tr["separate_normalization_calls_saved"] == tr["entries"]
