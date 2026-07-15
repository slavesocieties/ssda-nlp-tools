"""Computationally extracts content from normalized text.

Uses natural language instructions and manually created examples
to construct a conversation history that is then passed to an LLM
for chat completion via an api. The api should respond with a str
representation of a json document containing extracted content.
"""

from openai import OpenAI    
import json
import os
import re
from utility import *
# from schema import schema

##CONSTANTS
RECIPROCAL_RELS = {"parent": "child", "child": "parent", 
                    "grandparent": "grandchild", "grandchild": "grandparent",
                    "enslaver": "slave", "slave": "enslaver", 
                    "indenturer": "indentured servant", "indentured servant": "indenturer",
                    "spouse": "spouse",
                    "godparent": "godchild", "godchild": "godparent"}
NULLABLE_PEOPLE_PROPS = ['rank', 'origin', 'ethnicity', 'age', 'legitimate', 'occupation', 'phenotype', 'free']
PEOPLE_PROPS = NULLABLE_PEOPLE_PROPS + ['id', 'name', 'titles', 'relationships']
EVENT_PROPS = ['type', 'principals', 'date']

# Main model setting. Change this value when you want to test another model, remembering to tune other parameter when using reasoning model
MODEL_NAME = 'gpt-5.4-nano'

# Reasoning/thinking setting, set to 'none'
REASONING_EFFORT = 'medium'

# Optional hard cap. Leave as None unless you want to force shorter outputs.
MAX_COMPLETION_TOKENS = None

ALLOWED_RELATIONSHIP_TYPES = sorted(RECIPROCAL_RELS.keys())

EXTRACTION_SYSTEM_PROMPT = """
You are a historical Spanish church-register information extraction assistant.

Goal:
Extract structured people and event data from one normalized transcription, following the same style as training_data.json.

Return format:
- Return exactly one JSON object.
- The root object must contain only two keys: "people" and "events".
- Do not return a full training-data example.
- Do not include "raw", "normalized", "type", "language", "country", "state", "city", "institution", "id", or "entry" at the root.
- Do not include Markdown, code fences, explanations, or comments.

People rules:
1. Create one person object for each person explicitly mentioned in the transcription.
2. Use stable IDs in order of first appearance: P01, P02, P03, ...
3. Every person must have "id" and "name".
4. Use "titles" as a list, for example ["Don"] or ["Doña"]. If none, use [] or omit; the script will fill missing fields.
5. Use "relationships" as a list of objects with only:
   {"related_person": "Pxx", "relationship_type": "..."}.
6. Allowed relationship_type values are:
   parent, child, grandparent, grandchild, enslaver, slave,
   indenturer, indentured servant, spouse, godparent, godchild.
7. Relationships should be reciprocal when clearly stated or logically required.
   Example: if child has parent P02, P02 should have child relationship to the child.
8. Do not invent unnamed people. For "padre no conocido" / "father unknown", do not create a person.
9. Preserve names from the normalized text. Do not add surnames or expand initials unless clearly present.
10. Do not emit Unicode control characters, null characters, replacement characters, or padding characters.

Property rules:
- rank: military/social rank when explicit, otherwise null/omit.
- origin: place of origin/naturality when explicit, otherwise null/omit.
- ethnicity: ethnic/national origin words such as criolla, Carabalí, Angola, Guinea when used as ethnicity/origin context.
- age: use "infant", "adult", or a textual age if explicit.
- legitimate: true for hijo/hija legítimo/a; false for hijo/hija natural or padre no conocido; otherwise null/omit.
- occupation: use broad labels such as "cleric", "engineer", "soldier", "servant" when explicit.
- phenotype: color/status descriptors such as negro, moreno, pardo, blanco when applied to a person.
- free: true for libre/free; false for esclavo/esclava; otherwise null/omit.

Event rules:
1. Always include a baptism event for baptism records.
2. Include a birth event only when a birth date is explicit or inferable from phrases like "nació el día...".
3. Event principals must refer to person IDs.
4. Baptism event principal is the baptized person.
5. Birth event principal is the child.
6. Marriage event principals are the two spouses.
7. Dates should be ISO format YYYY-MM-DD when possible.
8. If the year is referred to as "presente año", "último", "dicho año", infer it from the baptism/marriage date when clear.
9. If the date cannot be resolved, use null rather than inventing.

Language normalization for extracted fields:
- The output JSON should use English-normalized values for the fields "age" and "origin".
- Do not translate person names.
- Do not translate titles unless the training examples clearly do so.
- Do not translate the full normalized transcription; only translate selected extracted field values.

Age rules:
- Translate age category words into English.
- "párvulo", "párvula", "niño", "niña", "criatura", or similar infant/child baptism terms should usually be "infant" unless the text gives a specific numeric age.
- "adulto" or "adulta" should be "adult".
- If a specific age is stated, preserve the number but use English units, e.g. "thirteen years old", "six months old".
- If age is not stated or not clearly implied, use null.

Origin rules:
- Translate common geographic descriptors into English when they are not part of a proper name.
- "natural de Guinea" -> "Guinea"
- "natural de la Costa de Guinea" -> "Coast of Guinea"
- "natural de Africa" -> "Africa"
- "natural de Nueva York" -> "New York"
- "natural de esta feligresía" -> "this parish"
- "natural de esta ciudad" -> "this city"
- "natural de esta" -> "this place" if the specific place cannot be resolved.
- Preserve proper place names unless there is a standard English form.
- If origin is not stated, use null.

Accuracy rules:
- Prefer omission/null over hallucination.
- Extract only facts grounded in the normalized transcription.
- Match the conventions of the few-shot examples from training_data.json.
""".strip()

EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_church_record_data",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "titles": {"type": "array", "items": {"type": "string"}},
                            "rank": {"type": ["string", "null"]},
                            "origin": {"type": ["string", "null"]},
                            "ethnicity": {"type": ["string", "null"]},
                            "age": {"type": ["string", "null"]},
                            "legitimate": {"type": ["boolean", "string", "null"]},
                            "occupation": {"type": ["string", "null"]},
                            "phenotype": {"type": ["string", "null"]},
                            "free": {"type": ["boolean", "string", "null"]},
                            "relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "related_person": {"type": "string"},
                                        "relationship_type": {
                                            "type": "string",
                                            "enum": [
                                                "parent", "child", "grandparent", "grandchild",
                                                "enslaver", "slave", "indenturer", "indentured servant",
                                                "spouse", "godparent", "godchild"
                                            ],
                                        },
                                    },
                                    "required": ["related_person", "relationship_type"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["id", "name"],
                        "additionalProperties": True,
                    },
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "principals": {"type": "array", "items": {"type": "string"}},
                            "date": {"type": ["string", "null"]},
                        },
                        "required": ["type", "principals"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["people", "events"],
            "additionalProperties": False,
        },
    },
}

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
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



def sanitize_string_value(value: str):
    cleaned = CONTROL_CHAR_RE.sub("", value)
    return cleaned, cleaned != value


def sanitize_extracted_payload(data, entry_id: str, log_path: str):
    changed_paths = []

    def clean_value(value, path="root"):
        if isinstance(value, str):
            cleaned, changed = sanitize_string_value(value)
            if changed:
                changed_paths.append(path)
            return cleaned

        if isinstance(value, list):
            return [clean_value(item, f"{path}[{i}]") for i, item in enumerate(value)]

        if isinstance(value, dict):
            return {key: clean_value(val, f"{path}.{key}") for key, val in value.items()}

        return value

    cleaned_data = clean_value(data)

    if changed_paths:
        debug_dir = "suspicious_extraction_outputs"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"{entry_id}_control_characters.json")

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log_failure(
            entry_id,
            log_path,
            "control_characters",
            "Removed unexpected control characters from: " + ", ".join(changed_paths),
            json.dumps(data, ensure_ascii=False),
        )

    return cleaned_data, bool(changed_paths)

def extract_data_from_volume(volume_record_path, instructions_path, training_data_path, keywords = None, match_mode = "or", max_shots = 1000, output_path = None, log_path = "failure_log.json"):
    """Extracts content from a series of transcribed entries from a historical document.

    Args:
        volume_record_path: either path to a json file containing a volume record or
        a dictionary containing this data

        instructions_path (str): path to a json file containing natural language instructions
        that will be passed to the llm as system messages

        training_data_path (str): path to a json file containing manually constructed examples
        of content extraction that will be used to train the llm

        keywords (list, optional): list of keywords that define a subset of training
        data to use in conjunction with the next parameter; if not included, all available
        training data will be used

        match_mode (str, optional): either `and` or `or`, defines subset of training data to use
        in conjunction with previous parameter

        max_shots (int, optional): defines maximum number of examples to include in conversation
        history supplied to llm
        
        out_path (str, optional): path to output volume record with extracted content to; volume
        record will not be saved if this is not included

        log_prefix (str, optional): a directory to log failed relationships (ending with /)

    Returns:
        Dict containing volume record and extracted content. 
    """
    data, volume_metadata = parse_volume_record(volume_record_path)

    examples = generate_training_data(training_data_path, keywords, match_mode=match_mode, max_shots=max_shots)
    instructions = collect_instructions(instructions_path, volume_metadata, "extraction")
    
    for x, entry in enumerate(data["entries"]):
        info = extract_data_from_entry(entry, volume_metadata, examples, instructions, log_path=log_path)

        if len(info) > 0:
            data["entries"][x]["data"] = json.loads(info)    
    
    if output_path != None:    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def extract_data_from_entry(entry, volume_metadata, examples, instructions, log_fails=True, log_path="failure_log.json"):
    """Extracts content from a single normalized entry from a historical document."""
    client = OpenAI()

    record_type = parse_record_type(volume_metadata)

    conversation = []

    # Strong task-specific system prompt first.
    conversation.append(
        {
            "role": "system",
            "content": EXTRACTION_SYSTEM_PROMPT
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
                    f"Extract the people and events JSON from this "
                    f"{example.get('language', volume_metadata['language'])} {example.get('type', record_type)} transcription. "
                    "Return only the data object with people and events:\n"
                    + example["normalized"]
                )
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": json.dumps(example["data"], ensure_ascii=False)
            }
        )

    # New query. Use metadata instead of hard-coded "Spanish baptismal register".
    conversation.append(
        {
            "role": "user",
            "content": (
                f"Extract the people and events JSON from this "
                f"{volume_metadata['language']} {record_type} transcription. "
                "Return only the data object with people and events:\n"
                + entry["normalized"]
            )
        }
    )

    completion_kwargs = {
        "model": MODEL_NAME,
        "response_format": EXTRACTION_RESPONSE_FORMAT,
        "messages": conversation,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if REASONING_EFFORT is not None:
        completion_kwargs["reasoning_effort"] = REASONING_EFFORT

    if MAX_COMPLETION_TOKENS is not None:
        completion_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS

    response = client.chat.completions.create(**completion_kwargs)

    init_result, usage = collect_streaming_response(response)

    if usage:
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
        thinking_part = f", reasoning={reasoning_tokens}" if reasoning_tokens is not None else ""
        print(
            f"[EXTRACT] entry={entry.get('id')} "
            f"prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
            f"{thinking_part}"
        )

    if log_fails:
        # log invalid jsons
        if not check_valid_json(init_result, entry['id'], log_path):
            return ""

        # Remove unexpected control/null characters before normal validation.
        parsed_result = json.loads(init_result)
        cleaned_result, _ = sanitize_extracted_payload(parsed_result, entry["id"], log_path)
        init_result = json.dumps(cleaned_result, ensure_ascii=False)

        # log invalid people
        people_result, people_valid = log_failed_people(init_result, entry['id'], log_path)

        # log invalid events
        event_result, event_valid = log_failed_events(people_result, entry['id'], log_path)

        # log invalid relationships
        rel_result, rel_valid = log_failed_relationships(event_result, entry['id'], log_path)

        # log dropped property information
        nulled_result, props_valid = fill_nulls(rel_result, entry['id'], log_path)

        # log relationship reciprocity fixing
        result, recip_valid = fix_relationships(nulled_result, entry['id'], log_path)
    else:
        result = init_result

    return result

def log_failure(failure_id, path, failure_type, failure_msg, original_data):
    """Logs a failure in the extraction process.

    Args:
        failure_id: the id of the failed entry. 

        path: path to failure log

        failure_type: type of failure (e.g. json, people, relationships, etc.)

        failure_msg: a message describing the failure

        original_data: the output that triggered the failure
    """    
    print(f"Logging {failure_type} failure for {failure_id}: {failure_msg}")
    if "/" in path and not os.path.exists(path.rsplit("/",1)[0]):
        os.makedirs(path.rsplit("/",1)[0])

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
        data["entries"] = []
        data["outputs"] = []
    
    if failure_id not in [x['id'] for x in data["outputs"]]:
        try:
            output = json.loads(original_data)
            data["outputs"].append(dict({"id": failure_id, "body": output}))
        except:
            data["outputs"].append(dict({"id": failure_id, "body": original_data}))
    
    data["entries"].append(dict({"id": failure_id, "type": failure_type, "message": failure_msg}))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_valid_json(data, id, path):
    """Checks if a json is formatted validly. Output fails this check if:
        - it is not valid Json format
        - it is missing the "people" property

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        True if the output is valid Json, False if not.
    """    
    json_str = fr'{data}'
    try:
        init_result = json.loads(json_str)
        if 'people' in init_result:
            return True
        else:
            log_failure(id, path , "json", f"Invalid json: missing 'people'", data)
            return False
    except:
        log_failure(id, path , "json", f"Invalid json", data)
        return False
    
def log_failed_people(data, id, path):
    """Checks if the "people" property has errors, including
        - a person is missing the property "id"
        - a person is missing the property "name"

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """    
    json_str = fr'{data}'
    data = json.loads(json_str)
    
    success = True
    valid_people = []
    for p in data['people']:
        if 'id' not in p or 'name' not in p:
            log_failure(id, path , "people", f"Missing 'name' or 'id' property", data)
            success = False
        else:
            valid_people.append(p)

    data['people'] = valid_people
        
    return json.dumps(data, ensure_ascii=False), success

def log_failed_events(data, id, path):
    """Checks if the "events" property has errors, including
        - an event is missing the property "type"
        - an event is missing the property "principals"
        - an event has invalid principals

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """    
    json_str = fr'{data}'
    data = json.loads(json_str)
    
    success = True
    if 'events' in data:
        valid_events = []
        for e in data['events']:
            if 'type' not in e:
                success=False
                log_failure(id, path , "events", f"Missing 'type' property", data)
            elif 'principals' not in e:
                success=False
                log_failure(id, path , "events", f"Missing 'principals' property", data)
            else:
                valid_events.append(e)

        data['events'] = valid_events
        
        ids = [p['id'] for p in data['people']]

        for e in data['events']:
            if isinstance(e['principals'], list):
                valid_principals = []
                for p in e['principals']:
                    if p not in ids:
                        success=False
                        log_failure(id, path , "events", f"Invalid principal: {p}", data)
                    else:
                        valid_principals.append(p)
                e['principals'] = valid_principals

    return json.dumps(data, ensure_ascii=False), success

def log_failed_relationships(data, id, path):
    """Checks for errors in the "relationships" properties, including
        - a relationship is missing the property "related_person"
        - a relationship is missing the property "relationship_type"
        - a relationship has an invalid related person
        - a relationship has an invalid relationship type
        - the relationship has unexpected properties
        Does NOT check for relationship reciprocity

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """   
    json_str = fr'{data}'
    data = json.loads(json_str)

    ids = [p['id'] for p in data['people']]

    success = True

    for p in [p for p in data['people'] if 'relationships' in p and isinstance(p['relationships'], list)]:
        valid_rels = []
        for r in [r for r in p['relationships'] if isinstance(r, dict)]:
            failure_msg = ""
            if 'related_person' not in r:
                failure_msg = "missing related person"
            elif 'relationship_type' not in r:
                failure_msg = "missing relationship type"
            elif r['relationship_type'] not in RECIPROCAL_RELS.keys():
                failure_msg = f"invalid relationship type {r['relationship_type']}"
            elif r['related_person'] not in ids:
                failure_msg = f"invalid related person {r['related_person']}"
            elif len(r.keys()) > 2:
                failure_msg = f"unexpected relationship properties: {r.keys()}"
            if len(failure_msg) > 0:
                success=False
                log_failure(id, path, "relationship",
                            f"Invalid relationship for {p['id']}: {failure_msg}", data)
            else:
                valid_rels.append(r)
        
        p['relationships'] = valid_rels

    return json.dumps(data, ensure_ascii=False), success

def fill_nulls(data, id, path):
    """Fills missing properties with nulls and empty lists,
        and checks for errors with properties, including:
        - the data type of a property is wrong
        - there is an unexpected property for a person or event

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """   
    json_str = fr'{data}'
    data = json.loads(json_str)

    success = True
    for person_index, p in enumerate(data['people']):
        for prop in NULLABLE_PEOPLE_PROPS:
            p[prop] = p.pop(prop) if prop in p else None
            if isinstance(p[prop], list):
                valid_values = [x for x in p[prop] if isinstance(x, str)]
                val = valid_values[0] if len(valid_values) > 0 else None
                log_failure(id, path, "people", 
                            f"Unexpected list property type for {p['id']}: {prop} = {val}", data)
                success = False
                p[prop] = val
        
        p['titles'] = p.pop('titles') if 'titles' in p else []
        if not isinstance(p['titles'], list):
            log_failure(id, path, "people", 
                        f"Titles of {p['id']} must be a list", data)
            success = False
            p['titles'] = [str(p['titles'])]
        
        p['relationships'] = p.pop('relationships') if 'relationships' in p else []
        if not isinstance(p['relationships'], list):
            log_failure(id, path, "relationships", 
                        f"Relationships of {p['id']} must be a list", data)
            success = False
            p['relationships'] = []
        
        valid_props = {}
        for k, v in p.items():
            if k not in PEOPLE_PROPS:
                log_failure(id, path, "people", 
                            f"Invalid property for person {p['id']}: {k} = {v}", data)
                success = False
            else:
                valid_props[k] = v
        data['people'][person_index] = valid_props

    if 'events' in data:
        valid_events = []
        for e in data['events']:
            if 'date' in e and not isinstance(e['date'], str):
                log_failure(id, path, "events", 
                            f"Invalid date", data)
                success = False
                e['date'] = None
            
            if 'principals' in e and not isinstance(e['principals'], list):
                log_failure(id, path, "events", 
                            f"Principals must be a list", data)
                success = False
                e['principals'] = [str(e['principals'])]

            valid_props = {}
            for k,v in e.items():
                if k not in EVENT_PROPS:
                    log_failure(id, path, "events", 
                                f"Invalid event property: {k} = {v}", data)
                    success=False
                else:
                    valid_props[k] = v
            e = valid_props

            if e['type'] == 'marriage' and len(e['principals']) != 2:
                log_failure(id, path, "events", 
                            f"Invalid event principals: marriage must have 2 principals.", data)
                success=False
            elif e['type'] == 'baptism' and len(e['principals']) != 1:
                log_failure(id, path, "events", 
                            f"Invalid event principals: baptism must have 1 principal.", data)
                success=False
            else:
                valid_events.append(e)
    
        data['events'] = valid_events

    return json.dumps(data, ensure_ascii=False), success

def fix_relationships(data, id, path):
    """Checks for irreciprocal relationship errors and fixes them with the following assumptions:
        - a principal is always the child, slave, godchild, or spouse in the relationship
        - irreciprocal relationships involving two non-principals are un-fixable
        - an irreciprocal relationship involving two unrelated relationship types 
            (eg child and slave) is un-fixable
        - if a relationship exists in one direction, it should exist in the other as well
        (unidirectional relationships are not dropped, assume a miss rather than a hallucination)

    Args:
        data: the output to check

        id: the id of the entry. 

        path: path to failure log

    Returns:
        A tuple (str, bool) where the first element is the data with errors removed, and the second
        is True if there were no errors, False if there were errors.
    """       
    json_str = fr'{data}'
    data = json.loads(json_str)

    success = True

    if not data.get("events"):
        log_failure(
            id,
            path,
            "events",
            "No valid events available; skipped relationship reciprocity fixing",
            data
        )
        return json.dumps(data, ensure_ascii=False), False
    
    relationships = {}
    for p in data['people']:
        relationships[p['id']] = {}
        for r in p['relationships']:
            relationships[p['id']][r['related_person']] = r['relationship_type']
    

    def del_relation(p1, p2):
        for person in data["people"]:
            if person["id"] == p1:
                person["relationships"][:] = [r for r in person["relationships"] if r["related_person"] != p2]

    def add_relation(p1, p2, type):
        del_relation(p1, p2)  # Ensure no duplicate relationships before adding
        for person in data["people"]:
            if person["id"] == p1:
                person["relationships"].append({"related_person": p2, "relationship_type": type})
                break  # Stop after modifying the first matching person

    def is_principal(p):
       return p in data['events'][0]['principals']

    for p1 in relationships.keys():
        for p2 in relationships[p1].keys():
            rel = relationships[p1][p2]
            if p1 not in relationships[p2]:
                success = False
                log_failure(id, path, "relationships", 
                            f"Non-reciprocal relationship for {p1} and {p2}: None and {rel}", data)
                add_relation(p2, p1, RECIPROCAL_RELS[rel])
            elif relationships[p2][p1] ==  RECIPROCAL_RELS[rel]:
                ##Valid reciprocal relationship
                pass
            elif relationships[p2][p1] !=  rel or (not is_principal(p1) and not is_principal(p2)):
                ##Give up
                success = False
                log_failure(id, path, "relationships", 
                            f"Unfixable non-reciprocal relationship for {p1} and {p2}: {relationships[p2][p1]} and {rel}", data)
                del_relation(p2, p1)
                del_relation(p1, p2)
            else:
                ##Fixable
                success = False
                log_failure(id, path, "relationships", 
                            f"Fixable non-reciprocal relationship for {p1} and {p2}: {relationships[p2][p1]} and {rel}", data)
                principal = p1 if is_principal(p1) else p2
                other = p2 if principal == p1 else p1
                if rel in ["child", "slave", "godchild"]:
                    add_relation(principal, other, RECIPROCAL_RELS[rel])
                    add_relation(other, principal, rel)
                else:
                    add_relation(other, principal, RECIPROCAL_RELS[rel])
                    add_relation(principal, other, rel)
                
    
    return json.dumps(data, ensure_ascii=False), success
