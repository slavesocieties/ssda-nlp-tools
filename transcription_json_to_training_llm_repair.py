import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Existing workflow modules
from extract import extract_data_from_volume
from normalize import normalize_volume
from utility import parse_volume_record
from convert_gemini_json_improved import convert_gemini_json


# =============================================================================
# User configuration
# =============================================================================


INPUT_FILENAME = "SSDA_0024_0034_Gemini_V2.json" # The raw output from Archivault
INSTRUCTIONS_FILENAME = "instructions.json" # The attached instuction files
TRAINING_DATA_FILENAME = "training_data.json" # The sample scheme we want
OUTPUT_FILENAME = "Generated_0024_0034_5.4nano_prompt.json" # Output file

# Metadata to attach to the generated volume record.
RECORD_TYPE = "baptism"
COUNTRY = "United States"
STATE = "Florida"
CITY = "San Agustin"
INSTITUTION = "parish"
TITLE = None
VOLUME_ID = None

MATCH_MODE = "or"
NORMALIZATION_MAX_SHOTS = 2
EXTRACTION_MAX_SHOTS = 5
TRAINING_KEYWORDS = {"type": RECORD_TYPE, "country": COUNTRY}

# Optional debugging outputs written next to the input file.
SAVE_INTERMEDIATE_VOLUME = False
SAVE_REPAIRED_VOLUME = False
SAVE_PROCESSED_VOLUME = False
INTERMEDIATE_VOLUME_FILENAME = "split_volume_before_llm_repair.json"
REPAIRED_VOLUME_FILENAME = "repaired_volume_after_llm.json"
PROCESSED_VOLUME_FILENAME = "processed_volume.json"
FAILURE_LOG_FILENAME = "failure_log.json"

# Input conversion
CONVERT_GEMINI_INPUT = True
CONVERTED_INPUT_FILENAME = "converted_input_for_llm_repair.json"

# LLM repair settings
USE_LLM_REPAIR = True # turn on to use LLM to repair the records
LLM_REPAIR_MODEL = "gpt-5.4-nano" # Change the model
LLM_REPAIR_TEMPERATURE = 1
LLM_REPAIR_MAX_WINDOW_PAGES = 2
LLM_REPAIR_OVERLAP_PAGES = 1
LLM_REPAIR_MAX_RETRIES = 2


# =============================================================================
# The content below is the trail to skip the normalized script
# =============================================================================



# If True, the LLM repair step also returns a normalized version of each repaired record.
# Then normalize_volume() will be skipped for entries already containing "normalized".
# Recommended first test: False, so we isolate repair first and keep existing normalize.py.
LLM_REPAIR_RETURNS_NORMALIZED = False

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# General utilities
# =============================================================================

def ensure_required_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def load_transcription_json(json_file_path: Path) -> List[Dict[str, Any]]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to be a list of items.")
    return data


def infer_volume_id(data: List[Dict[str, Any]], fallback: Optional[int] = None) -> int:
    if fallback is not None:
        return fallback

    for item in data:
        metadata = item.get("metadata", {})
        identifier = metadata.get("identifier")
        if identifier is not None:
            try:
                return int(identifier)
            except (TypeError, ValueError):
                pass

        for image in item.get("images", []):
            file_name = image.get("file", "")
            match = re.match(r"(\d+)-\d+\.jpg$", file_name)
            if match:
                return int(match.group(1))

    raise ValueError("Could not infer volume id from the transcription JSON.")


def infer_title(data: List[Dict[str, Any]], fallback: Optional[str] = None) -> str:
    if fallback:
        return fallback

    for item in data:
        metadata = item.get("metadata", {})
        titles = metadata.get("title")
        if isinstance(titles, list) and titles:
            return str(titles[0])
        if isinstance(titles, str) and titles.strip():
            return titles.strip()

        direct_title = item.get("title")
        if isinstance(direct_title, list) and direct_title:
            return str(direct_title[0])
        if isinstance(direct_title, str) and direct_title.strip():
            return direct_title.strip()

    return "Untitled volume"


def prepare_input_file(input_path: Path) -> Path:
    """Optionally convert Gemini-style input into legacy-compatible format."""
    if not CONVERT_GEMINI_INPUT:
        return input_path

    converted_path = BASE_DIR / CONVERTED_INPUT_FILENAME
    success = convert_gemini_json(str(input_path), str(converted_path))
    if not success:
        raise RuntimeError(f"Failed to convert Gemini input file: {input_path}")
    return converted_path


# =============================================================================
# Page-level volume construction
# =============================================================================

def image_id_from_file_name(file_name: str) -> str:
    # 239746-0017.jpg -> 0017
    stem = Path(file_name).stem
    return stem.split("-")[-1]


def build_volume_record_from_pages(
    json_file_path: Path,
    volume_id: Optional[int] = None,
    record_type: str = "baptism",
    country: str = "United States",
    state: str = "Florida",
    city: str = "San Agustin",
    institution: str = "parish",
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a page-level volume record.

    Unlike the old script, this does NOT try to split entries with language-specific
    weekday/month rules. Each page becomes one page object, and the LLM repair step
    is responsible for segmentation, merging broken records, and dropping junk.
    """
    data = load_transcription_json(json_file_path)
    volume_id = infer_volume_id(data, fallback=volume_id)
    title = infer_title(data, fallback=title)

    pages: List[Dict[str, str]] = []

    for item in data:
        for image in item.get("images", []):
            transcription = image.get("transcription", "")
            file_name = image.get("file", "")
            if not transcription or not file_name:
                continue

            page_id = image_id_from_file_name(file_name)
            pages.append(
                {
                    "page_id": page_id,
                    "file": file_name,
                    "raw": transcription.strip(),
                    "cleaned_raw": clean_text(transcription),
                }
            )

    return {
        "type": record_type,
        "country": country,
        "state": state,
        "city": city,
        "institution": institution,
        "id": volume_id,
        "title": title,
        "pages": pages,
        "entries": [],
    }


# =============================================================================
# LLM repair
# =============================================================================

def collect_streaming_response(response) -> Tuple[str, Optional[Any]]:
    full_text = ""
    usage = None

    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                full_text += delta.content

        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage

    return full_text, usage


def build_repair_prompt(
    volume_record: Dict[str, Any],
    pages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    record_type = volume_record.get("type", "record")

    system_text = f"""
You are repairing segmentation for historical handwritten {record_type} records.

Your task:
1. Read the page transcriptions.
2. Identify complete logical records.
3. Merge records split across page boundaries.
4. Remove page headers, page numbers, years, decorative marks, marginal names, and obvious junk.
5. Preserve the original text as much as possible in "raw".
6. Do not extract people/events. Do not invent missing content.
7. If a record clearly starts inside the provided window but continues after the last page in the window, include it with status="incomplete_trailing".
8. If text is only the ending of a record that started before this window and cannot be merged with a start inside this window, put it in dropped_fragments.
9. Return valid JSON only.

Important:
- Do NOT rely on Spanish weekday names only. The book may be in any language.
- Use structural cues: numbering, repeated formula, date line, officiant phrase, event verb, naming phrase, godparent/witness phrase, closing/signature.
- A complete record usually has a beginning/date/formula and an ending/signature/closing phrase, but wording can vary.
- If a record starts on page A and ends on page B, output ONE merged entry.
""".strip()

    if LLM_REPAIR_RETURNS_NORMALIZED:
        normalized_instruction = (
            'For each entry, also return "normalized": a lightly normalized version of the repaired record. '
            "Keep meaning unchanged. Normalize spacing, obvious line-break hyphenation, and punctuation only."
        )
    else:
        normalized_instruction = 'Do not normalize. Set "normalized" to an empty string.'

    user_payload = {
        "volume_metadata": {
            "type": volume_record.get("type"),
            "country": volume_record.get("country"),
            "state": volume_record.get("state"),
            "city": volume_record.get("city"),
            "institution": volume_record.get("institution"),
            "title": volume_record.get("title"),
        },
        "id_guidance": {
            "rule": "For each repaired entry, assign a temporary id using the page where the record begins, e.g. 0017-01.",
            "note": "Final ids will be re-numbered by the script; focus on correct source_pages and raw text.",
        },
        "output_schema": {
            "entries": [
                {
                    "id": "page-entry id such as 0017-01",
                    "source_pages": ["0017", "0018"],
                    "status": "complete or incomplete_trailing",
                    "raw": "complete repaired record text",
                    "normalized": "empty string unless normalization was requested",
                }
            ],
            "dropped_fragments": [
                {
                    "source_page": "0017",
                    "raw": "dropped header, junk, or unmergeable fragment",
                    "reason": "short explanation",
                }
            ],
        },
        "normalization_instruction": normalized_instruction,
        "pages": [
            {
                "page_id": p["page_id"],
                "file": p["file"],
                "transcription": p["raw"],
            }
            for p in pages
        ],
    }

    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": "Repair the following page transcriptions and return JSON only:\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
    ]


def parse_llm_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def validate_repair_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    entries = payload.get("entries", [])
    dropped = payload.get("dropped_fragments", [])

    if not isinstance(entries, list):
        entries = []
    if not isinstance(dropped, list):
        dropped = []

    clean_entries = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        raw = str(e.get("raw", "")).strip()
        if not raw:
            continue

        entry_id = str(e.get("id", "")).strip()
        source_pages = e.get("source_pages", [])
        if not isinstance(source_pages, list):
            source_pages = [str(source_pages)]

        status = str(e.get("status", "complete")).strip() or "complete"
        normalized = str(e.get("normalized", "") or "")

        clean_entries.append(
            {
                "id": entry_id,
                "source_pages": [str(x) for x in source_pages if str(x).strip()],
                "status": status,
                "raw": raw,
                "normalized": normalized,
            }
        )

    clean_dropped = []
    for d in dropped:
        if not isinstance(d, dict):
            continue
        raw = str(d.get("raw", "")).strip()
        if not raw:
            continue
        clean_dropped.append(
            {
                "source_page": str(d.get("source_page", "")),
                "raw": raw,
                "reason": str(d.get("reason", "")),
            }
        )

    return {"entries": clean_entries, "dropped_fragments": clean_dropped}


def repair_page_window_with_llm(
    volume_record: Dict[str, Any],
    pages: List[Dict[str, str]],
    client: OpenAI,
) -> Dict[str, Any]:
    messages = build_repair_prompt(volume_record, pages)
    last_error = None

    for attempt in range(LLM_REPAIR_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_REPAIR_MODEL,
                messages=messages,
                reasoning_effort="medium", # Very important, delete when not using reasoning model, otherwise the script will not work
                response_format={"type": "json_object"},
                temperature=LLM_REPAIR_TEMPERATURE,
                stream=True,
                stream_options={"include_usage": True},
            )
            text, usage = collect_streaming_response(response)

            if usage:
                page_ids = ",".join(p["page_id"] for p in pages)
                print(
                    f"[LLM_REPAIR] pages={page_ids} "
                    f"prompt={usage.prompt_tokens}, "
                    f"completion={usage.completion_tokens}, "
                    f"total={usage.total_tokens}"
                )

            payload = parse_llm_json_response(text)
            return validate_repair_payload(payload)

        except Exception as exc:
            last_error = exc
            print(f"[LLM_REPAIR] attempt {attempt + 1} failed: {exc}")

    raise RuntimeError(f"LLM repair failed after retries: {last_error}")


def window_pages(pages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """
    Create overlapping page windows.

    Default window size 2, overlap 1:
    [0013,0014], [0014,0015], [0015,0016], ...

    The script deduplicates entries after repair.
    """
    if not pages:
        return []

    size = max(1, LLM_REPAIR_MAX_WINDOW_PAGES)
    overlap = max(0, min(LLM_REPAIR_OVERLAP_PAGES, size - 1))
    step = size - overlap

    windows = []
    i = 0
    while i < len(pages):
        windows.append(pages[i : i + size])
        if i + size >= len(pages):
            break
        i += step
    return windows


def entry_sort_key(entry: Dict[str, Any]) -> Tuple[str, int, str]:
    entry_id = str(entry.get("id", ""))
    m = re.match(r"^(\d{4})-(\d{2,3})$", entry_id)
    if m:
        return (m.group(1), int(m.group(2)), entry_id)

    source_pages = entry.get("source_pages") or []
    first_page = str(source_pages[0]) if source_pages else "9999"
    return (first_page, 999, entry_id)


def deduplicate_repaired_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Overlapping windows may return the same complete entry twice.
    Deduplicate using the first 500 normalized characters of raw text.
    """
    seen = set()
    unique = []

    for e in sorted(entries, key=entry_sort_key):
        raw_key = re.sub(r"\s+", " ", e.get("raw", "").lower()).strip()[:500]
        if raw_key in seen:
            continue
        seen.add(raw_key)
        unique.append(e)

    return unique


def normalize_repaired_entry_ids(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-number entries by their first source page.

    Example:
    source_pages ["0017", "0018"] -> 0017-01 / 0017-02 depending order.
    """
    sorted_entries = sorted(entries, key=entry_sort_key)
    counts: Dict[str, int] = {}

    final_entries = []
    for e in sorted_entries:
        source_pages = e.get("source_pages") or []
        first_page = str(source_pages[0]) if source_pages else None

        if not first_page:
            m = re.match(r"^(\d{4})-", str(e.get("id", "")))
            first_page = m.group(1) if m else "0000"

        counts[first_page] = counts.get(first_page, 0) + 1
        new_id = f"{first_page}-{counts[first_page]:02d}"

        new_entry = {
            "id": new_id,
            "raw": e["raw"],
        }

        if LLM_REPAIR_RETURNS_NORMALIZED and e.get("normalized"):
            new_entry["normalized"] = e["normalized"]

        final_entries.append(new_entry)

    return final_entries


def repair_entries_with_llm(volume_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main LLM structural repair step.

    Input:
        volume_record with page-level "pages"

    Output:
        volume_record with repaired entry-level "entries"
    """
    pages = volume_record.get("pages", [])
    if not pages:
        raise ValueError("volume_record has no pages to repair.")

    client = OpenAI()

    all_repaired_entries: List[Dict[str, Any]] = []
    all_dropped_fragments: List[Dict[str, Any]] = []

    for page_window in window_pages(pages):
        payload = repair_page_window_with_llm(
            volume_record=volume_record,
            pages=page_window,
            client=client,
        )

        all_repaired_entries.extend(payload["entries"])
        all_dropped_fragments.extend(payload["dropped_fragments"])

    unique_entries = deduplicate_repaired_entries(all_repaired_entries)
    final_entries = normalize_repaired_entry_ids(unique_entries)

    updated = dict(volume_record)
    updated["entries"] = final_entries
    updated["llm_repair_dropped_fragments"] = all_dropped_fragments
    return updated


# =============================================================================
# Optional normalization wrapper
# =============================================================================

def normalize_volume_if_needed(
    volume_record: Dict[str, Any],
    instructions_path: str,
    training_data_path: str,
) -> Dict[str, Any]:
    """
    If LLM repair produced normalized text for every entry, skip normalize.py.
    Otherwise use the existing normalize_volume().
    """
    entries = volume_record.get("entries", [])
    has_all_normalized = bool(entries) and all(
        isinstance(e.get("normalized"), str) and e.get("normalized", "").strip()
        for e in entries
    )

    if has_all_normalized:
        print("[NORMALIZE] skipped because LLM repair already returned normalized text.")
        return volume_record

    return normalize_volume(
        volume_record,
        instructions_path,
        training_data_path,
        keywords=TRAINING_KEYWORDS,
        match_mode=MATCH_MODE,
        max_shots=NORMALIZATION_MAX_SHOTS,
    )


# =============================================================================
# Final schema conversion
# =============================================================================

def convert_processed_volume_to_training_data(processed_volume: Dict[str, Any]) -> Dict[str, Any]:
    parsed_volume, volume_metadata = parse_volume_record(processed_volume)
    language = volume_metadata["language"]

    examples: List[Dict[str, Any]] = []
    for entry in processed_volume.get("entries", []):
        examples.append(
            {
                "type": volume_metadata.get("type", ""),
                "language": language,
                "country": parsed_volume.get("country", ""),
                "state": volume_metadata.get("state", ""),
                "city": parsed_volume.get("city", ""),
                "institution": volume_metadata.get("institution", ""),
                "id": volume_metadata.get("id", ""),
                "entry": entry.get("id", ""),
                "raw": entry.get("raw", ""),
                "normalized": entry.get("normalized", ""),
                "data": entry.get("data", {}),
            }
        )

    return {"examples": examples}


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline() -> Path:
    raw_input_path = BASE_DIR / INPUT_FILENAME
    instructions_path = BASE_DIR / INSTRUCTIONS_FILENAME
    training_data_path = BASE_DIR / TRAINING_DATA_FILENAME
    output_path = BASE_DIR / OUTPUT_FILENAME
    failure_log_path = BASE_DIR / FAILURE_LOG_FILENAME

    ensure_required_file(raw_input_path, "input transcription JSON")
    ensure_required_file(instructions_path, "instructions file")
    ensure_required_file(training_data_path, "training data file")

    input_path = prepare_input_file(raw_input_path)

    volume_record = build_volume_record_from_pages(
        json_file_path=input_path,
        volume_id=VOLUME_ID,
        record_type=RECORD_TYPE,
        country=COUNTRY,
        state=STATE,
        city=CITY,
        institution=INSTITUTION,
        title=TITLE,
    )

    if SAVE_INTERMEDIATE_VOLUME:
        save_json(volume_record, BASE_DIR / INTERMEDIATE_VOLUME_FILENAME)

    if USE_LLM_REPAIR:
        volume_record = repair_entries_with_llm(volume_record)
    else:
        raise RuntimeError("This script is designed for LLM repair. Set USE_LLM_REPAIR=True.")

    if SAVE_REPAIRED_VOLUME:
        save_json(volume_record, BASE_DIR / REPAIRED_VOLUME_FILENAME)

    normalized_volume = normalize_volume_if_needed(
        volume_record,
        str(instructions_path),
        str(training_data_path),
    )

    processed_volume = extract_data_from_volume(
        normalized_volume,
        str(instructions_path),
        str(training_data_path),
        keywords=TRAINING_KEYWORDS,
        match_mode=MATCH_MODE,
        max_shots=EXTRACTION_MAX_SHOTS,
        output_path=str(BASE_DIR / PROCESSED_VOLUME_FILENAME) if SAVE_PROCESSED_VOLUME else None,
        log_path=str(failure_log_path),
    )

    training_data = convert_processed_volume_to_training_data(processed_volume)
    save_json(training_data, output_path)
    return output_path


def main() -> None:
    output_path = run_pipeline()
    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()
