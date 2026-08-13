"""Submit one capped Sonnet evaluation request to Anthropic's async Batch API."""
import json, os, urllib.request
from pathlib import Path
from ssda_nlp_tools.batch_extract import build_messages, plan_batches

root = Path(__file__).resolve().parent
examples = json.loads((root / "training_data.json").read_text(encoding="utf-8"))["examples"]
source = json.loads((root / "Sample_output/Generated_0035_0044_4o_prompt_V2.json").read_text(encoding="utf-8"))["examples"]
entries = [{"entry": e["entry"], "raw": e.get("raw") or e.get("normalized") or ""} for e in source[:8]]
messages = build_messages(entries, examples, [])
system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
turns = [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
         for m in messages if m["role"] != "system"]
payload = {"requests": [{"custom_id": "sonnet-heldout-b0", "params": {
    "model": "claude-sonnet-5", "max_tokens": 10000, "system": system, "messages": turns
}}]}
key = os.environ["ANTHROPIC_API_KEY"]
req = urllib.request.Request("https://api.anthropic.com/v1/messages/batches",
    data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
    headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    job = json.loads(r.read().decode())
(root / "sonnet_batch_job.json").write_text(json.dumps({"id": job.get("id"), "processing_status": job.get("processing_status")}, indent=2), encoding="utf-8")
print(json.dumps({"id": job.get("id"), "processing_status": job.get("processing_status")}))
