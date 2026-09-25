"""run_live_test.py output must feed the QA tools directly (no reshaping)."""
import importlib.util
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("run_live_test", os.path.join(ROOT, "run_live_test.py"))
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)

from ssda_nlp_tools.qa import qa_volume  # noqa: E402


class _FakeCompletions:
    def create(self, model, messages, **_):
        ids = [e["entry"] for e in json.loads(messages[-1]["content"])["entries"]]
        content = json.dumps({"results": [
            {"entry": i, "normalized": "n", "data": {"people": [], "events": []}} for i in ids]})
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                      prompt_tokens_details=None)
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=msg)])


def test_live_output_uses_entries_and_is_read_by_qa(tmp_path, monkeypatch):
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions()))
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    tail = {"role": "user", "content": json.dumps({"entries": [
        {"entry": "239746-0013-01", "transcription": "a"},
        {"entry": "239746-0013-02", "transcription": "b"}]})}
    batch = tmp_path / "239746.batches.jsonl"
    batch.write_text(json.dumps({"header": {"volume": "239746", "model": "gpt-5.6-luna",
                                            "prefix_messages": []}}) + "\n"
                     + json.dumps({"custom_id": "239746-b0000", "tail_message": tail}) + "\n",
                     encoding="utf-8")
    out = tmp_path / "live.json"
    assert live.main([str(batch), "--confirm", "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert "records" not in result
    assert [e["id"] for e in result["entries"]] == ["239746-0013-01", "239746-0013-02"]
    assert qa_volume(str(out))["entries"] == 2
