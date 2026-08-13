"""Submit one capped Gemini evaluation request to the asynchronous Batch API."""
import argparse, json, os, urllib.request
from pathlib import Path
from ssda_nlp_tools.batch_extract import build_messages

ap = argparse.ArgumentParser()
ap.add_argument("model")
args = ap.parse_args()
root = Path(__file__).resolve().parent
examples = json.loads((root / "training_data.json").read_text(encoding="utf-8"))["examples"]
source = json.loads((root / "Sample_output/Generated_0035_0044_4o_prompt_V2.json").read_text(encoding="utf-8"))["examples"]
entries = [{"entry": e["entry"], "raw": e.get("raw") or e.get("normalized") or ""} for e in source[:8]]
messages = build_messages(entries, examples, [])
system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"]
request = {"system_instruction": {"parts": [{"text": system}]}, "contents": contents,
           "generation_config": {"temperature": 0, "max_output_tokens": 10000,
                                 "response_mime_type": "application/json"}}
payload = {"batch": {"display_name": f"ssda-heldout-{args.model}", "input_config": {
    "requests": {"requests": [{"request": request, "metadata": {"key": "heldout-b0"}}]}
}}}
key = os.environ["GEMINI_API_KEY"]
req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:batchGenerateContent",
    data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
    headers={"x-goog-api-key": key, "content-type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    job = json.loads(r.read().decode())
out = {"name": job.get("name"), "state": job.get("state")}
(root / f"{args.model}_batch_job.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out))
