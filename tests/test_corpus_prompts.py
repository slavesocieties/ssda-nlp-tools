"""Offline tests for the corpus->batches bridge (run_corpus_prompts.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(argv):
    sys.path.insert(0, ROOT)
    import run_corpus_prompts
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return run_corpus_prompts.main(argv)
    finally:
        os.chdir(cwd)


def _mini_corpus(tmp_path):
    """Build a one-volume segmented corpus from the in-repo raw pages, so these
    tests do not depend on the (gitignored, regenerable) full out_corpus/."""
    from ssda_nlp_tools.segment import load_pages, segment_volume
    pages = load_pages(os.path.join(ROOT, "Text data/SSDA_0013_0023_Gemini_V2.json"))
    res = segment_volume(pages)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with open(corpus / "239746.segmented.json", "w", encoding="utf-8") as f:
        json.dump({"volume": "239746", "stats": res["stats"],
                   "entries": res["entries"]}, f, ensure_ascii=False)
    return str(corpus)


def test_compact_write_and_expand_roundtrip(tmp_path):
    corpus = _mini_corpus(tmp_path)
    out = str(tmp_path / "b")
    assert _run(["--corpus", corpus, "--limit", "1", "--outdir", out]) == 0

    files = [f for f in os.listdir(out) if f.endswith(".batches.jsonl")]
    assert len(files) == 1
    path = os.path.join(out, files[0])

    with open(path, encoding="utf-8") as fh:
        header = json.loads(fh.readline())["header"]
        rows = [json.loads(l) for l in fh]
    assert header["prefix_messages"][0]["role"] == "system"
    assert all(r["custom_id"] and r["tail_message"]["role"] == "user" for r in rows)

    # expand -> verbatim Batch API format
    assert _run(["--expand", path]) == 0
    api = path.replace(".batches.jsonl", ".batchapi.jsonl")
    lines = [json.loads(l) for l in open(api, encoding="utf-8")]
    assert len(lines) == len(rows)
    first = lines[0]
    assert first["url"] == "/v1/chat/completions"
    assert first["body"]["messages"] == header["prefix_messages"] + [rows[0]["tail_message"]]
    assert first["body"]["response_format"] == {"type": "json_object"}

    # manifest sanity
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert man["totals"]["calls"] == len(rows)
    assert man["totals"]["prefix_tokens_per_call"] > 5000   # real few-shot pool present


def _expand_body(tmp_path, sub, model, reasoning):
    # isolated corpus per call so the same tmp_path can be reused across levels
    from ssda_nlp_tools.segment import load_pages, segment_volume
    base = tmp_path / sub
    corpus = base / "corpus"
    corpus.mkdir(parents=True)
    pages = load_pages(os.path.join(ROOT, "Text data/SSDA_0013_0023_Gemini_V2.json"))
    res = segment_volume(pages)
    with open(corpus / "239746.segmented.json", "w", encoding="utf-8") as f:
        json.dump({"volume": "239746", "stats": res["stats"], "entries": res["entries"]},
                  f, ensure_ascii=False)
    out = str(base / "b")
    assert _run(["--corpus", str(corpus), "--limit", "1", "--outdir", out,
                 "--model", model, "--reasoning", reasoning]) == 0
    path = os.path.join(out, next(f for f in os.listdir(out) if f.endswith(".batches.jsonl")))
    assert _run(["--expand", path]) == 0
    api = path.replace(".batches.jsonl", ".batchapi.jsonl")
    return json.loads(open(api, encoding="utf-8").readline())["body"]


def test_reasoning_effort_pinned_into_openai_send_body(tmp_path):
    """The pinned reasoning level lands verbatim in the OpenAI send body, so the
    production run matches the level the quality numbers were measured at rather
    than the provider default."""
    for i, level in enumerate(("minimal", "low", "medium", "high")):
        body = _expand_body(tmp_path, f"r{i}_{level}", "gpt-5.6-luna", level)
        assert body["reasoning_effort"] == level
        assert body["model"] == "gpt-5.6-luna"


def test_reasoning_effort_omitted_for_non_openai(tmp_path):
    body = _expand_body(tmp_path, "r_claude", "claude-haiku-4.5", "high")
    assert "reasoning_effort" not in body     # Anthropic body must not carry it


def test_partial_entries_are_tagged_not_dropped(tmp_path):
    corpus = _mini_corpus(tmp_path)
    out = str(tmp_path / "b2")
    assert _run(["--corpus", corpus, "--limit", "1", "--outdir", out]) == 0
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    vol = next(iter(man["volumes"].values()))
    # entries count must equal the segmented file's entries (nothing silently dropped)
    seg = json.load(open(os.path.join(corpus, "239746.segmented.json"), encoding="utf-8"))
    n_nonempty = sum(1 for e in seg["entries"] if e.get("text", "").strip())
    assert vol["entries"] == n_nonempty
