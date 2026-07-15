"""Computationally normalizes raw transcribed text.

Uses natural language instructions and manually created examples
to construct a conversation history that is then passed to an LLM
for chat completion via an api. The api should respond with normalized
plain text only.
"""

import json
from utility import *
from openai import OpenAI


# Main model setting. Change this value when you want to test another model, remembering to tune other parameter when using reasoning model
MODEL_NAME = 'gpt-5.4-nano'

# Reasoning/thinking setting, set to 'none'
REASONING_EFFORT = 'medium'

# Optional hard cap. Leave as None unless you want to force shorter outputs.
MAX_COMPLETION_TOKENS = None


NORMALIZATION_SYSTEM_PROMPT = """
You are a historical Spanish church-register normalization assistant.

Goal:
Convert a raw transcription into a readable normalized transcription that follows the style of the provided training examples.

Return format:
- Return ONLY the normalized transcription as plain text.
- Do NOT return JSON.
- Do NOT wrap the answer in quotes.
- Do NOT use Markdown, code fences, bullets, labels, explanations, or commentary.

Normalization rules:
1. Preserve the original meaning and all factual content.
2. Preserve every person name, place name, title, occupation, date, and relationship phrase that appears in the raw text.
3. Do not invent missing words, names, dates, people, or relationships.
4. Expand obvious historical spelling/spacing issues when confidence is high.
   Examples: "diezyseis" -> "dieciséis"; "mil sete cientos" -> "mil setecientos"; broken line spacing should be repaired.
5. Fix obvious line-break hyphenation and spacing artifacts.
   Example: "Ciu dad" -> "ciudad"; "Ygle sia" -> "Yglesia" or "Iglesia" based on context.
6. Improve readability with sentence boundaries and punctuation, matching the training_data.json style.
7. Keep the record in Spanish. Do not translate into English.
8. Keep historical names close to the transcription. Add accents only when obvious from standard spelling or examples.
9. Preserve uncertainty. If a word/name is unclear, keep the visible transcription rather than guessing.
10. Remove page headers, marginalia, decorative marks, and obvious non-record junk only if they are clearly not part of the record.
11. Do not over-modernize. The output should be readable, not a rewritten modern summary.
12. Do not drop closing formulae such as "y lo firmé", witnesses/godparents, or officiant phrases.
13. Do not emit null characters, replacement characters, or padding characters.
""".strip()


def collect_streaming_response(response):
    full_text = ""
    usage = None

    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content

        if chunk.usage is not None:
            usage = chunk.usage

    return full_text, usage



def normalize_volume(volume_record_path, instructions_path, training_data_path, keywords = None, match_mode = "or", max_shots = 1000, output_path = None):
    """Normalizes text from a series of transcribed entries from a historical document."""
    data, volume_metadata = parse_volume_record(volume_record_path)

    examples = generate_training_data(training_data_path, keywords, match_mode=match_mode, max_shots=max_shots)
    instructions = collect_instructions(instructions_path, volume_metadata, "normalization")

    for x, entry in enumerate(data["entries"]):
        norm = normalize_entry(entry, volume_metadata, examples, instructions)
        data["entries"][x]["normalized"] = norm

    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def normalize_entry(entry, volume_metadata, examples, instructions):
    """Normalizes text from a single transcribed entry from a historical document."""
    client = OpenAI()    

    record_type = parse_record_type(volume_metadata)

    conversation = []

    # Strong task-specific system prompt first.
    conversation.append(
        {
            "role": "system",
            "content": NORMALIZATION_SYSTEM_PROMPT
        }
    )

    # Existing project instructions from instructions.json.
    for instruction in instructions:
        conversation.append(
            {
                "role": "system",
                "content": instruction["text"]
            }
        )

    # Few-shot examples from training_data.json.
    for example in examples:
        conversation.append(
            {
                "role": "user",
                "content": (
                    f"Normalize this {volume_metadata['language']} {record_type} transcription. "
                    "Return only normalized Spanish text:\n"
                    + example["raw"]
                )
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": example["normalized"]
            }
        )

    # New query.
    conversation.append(
        {
            "role": "user",
            "content": (
                f"Normalize this {volume_metadata['language']} {record_type} transcription. "
                "Follow the style of the examples. Return only normalized Spanish text:\n"
                + entry["raw"]
            )
        }
    )

    completion_kwargs = {
        "model": MODEL_NAME,
        "messages": conversation,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if REASONING_EFFORT is not None:
        completion_kwargs["reasoning_effort"] = REASONING_EFFORT

    if MAX_COMPLETION_TOKENS is not None:
        completion_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS

    response = client.chat.completions.create(**completion_kwargs)

    result, usage = collect_streaming_response(response)
    result = result.strip().strip("`").strip()

    if usage:
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
        thinking_part = f", reasoning={reasoning_tokens}" if reasoning_tokens is not None else ""
        print(
            f"[NORMALIZE] entry={entry.get('id')} "
            f"prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
            f"{thinking_part}"
        )

    return result


# normalize_volume("testing\\6517_sample.json", "instructions.json", "training_data.json", keywords={"language": "Spanish"}, match_mode="or", max_shots=1000, output_path="testing\\6517_sample.json")
