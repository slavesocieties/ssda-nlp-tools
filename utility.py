"""Utility functions to support other pipeline components.
"""

import json
from random import sample
from PIL import Image, ImageEnhance
import numpy as np
import boto3
import os

def manual_htr_training_data_generation(image_root="segmented", output_dir="htr_training_data.json"):
    data = {"images": []}
    deletes = []  
    for folder, subfolders, files in os.walk(image_root):
        for file in files:
            im = Image.open(os.path.join(folder, file))
            pix = im.size[0] * im.size[1]
            idx = file[:file.find(".")]
            col = "normalized"
            txt = input(f"Enter transcription for {idx}: ")
            if txt == "":
                deletes.append(os.path.join(folder, file))
                continue
            data["images"].append({"id": idx, "color": col, "text": txt, "pixels": pix})

    for file in deletes:
        os.unlink(file)

    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(data, f)

#manual_htr_training_data_generation(image_root="segmented", output_dir="htr_training_data.json")

def dynamic_binarization(image_path, output_path=None, binarization_quantile=0.1, verbose=False):
    im = Image.open(image_path)
    data = np.array(im)    
    im.close()
    bin_thresh = np.quantile(data, binarization_quantile)
    data = np.where(data <= bin_thresh, 0, 255)
    im = Image.fromarray(data).convert("L")

    if output_path is not None:
        im.save(output_path)
    else:
        im.save(image_path)
    
    if verbose:
        print(f"Dynamically binarized image saved to {output_path}")

#dynamic_binarization("segmented\\15834-0093-03-02.jpg", output_path="dyn_bin.jpg")

def enhance_contrast(image_path, output_path=None, factor=2.0, verbose=False):    
    image = Image.open(image_path).convert('L')  # Convert to grayscale if not already
    
    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(image)       
    enhanced_image = enhancer.enhance(factor)
    image.close()
    
    # Save the enhanced image
    if output_path is not None:
        enhanced_image.save(output_path)
    else:
        enhanced_image.save(image_path)
    
    if verbose:
        print(f"Enhanced image saved to {output_path}")

#enhance_contrast("segmented\\15834-0093-03-02.jpg", output_path="enhanced.jpg")

def check_binarized(path_to_image):
    with Image.open(path_to_image) as im:    
        if im.getbands()[0] == '1':        
            return "bin"
        arr = np.asarray(im)
        if np.std(arr) < 60:
            return "gray"
        return "semi-bin"

def load_volume_metadata(volume_id, volume_metadata_path = "volumes.json"):
    with open(volume_metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for volume in data:        
        if volume["fields"]["identifier"] == volume_id:
            if "Baptisms" in volume["fields"]["subject"]:
                volume["type"] = "baptism"
            elif "Marriages" in volume["fields"]["subject"]:
                volume["type"] = "marriage"
            elif "Burials" in volume["fields"]["subject"]:
                volume["type"] = "burial"
            return volume
        
    return None

def generate_block_htr_training_data(path_to_transcription_json, bucket_name="ssda-openai-test"):
    with open(path_to_transcription_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # TODO this needs to be refactored to allow for training data from multiple volumes
    volume_id = data["id"]
    examples = []

    for entry in data["entries"]:
        example = {"id": entry["id"], "lines": entry["lines"]}
        example["color"] = f"https://{bucket_name}.s3.amazonaws.com/{volume_id}-{entry['id']}-color.jpg"
        example["pooled"] = f"https://{bucket_name}.s3.amazonaws.com/{volume_id}-{entry['id']}-pooled.jpg"
        examples.append(example)

    return examples

def generate_htr_training_data(bucket_name="ssda-htr-training", metadata_path="volumes.json", keywords=None, match_mode="or", color=None, max_shots=10):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    volumes = []

    for volume in metadata:
        if keywords == None:
            volumes.append(volume["id"])
        elif match_mode == "or":            
            for key in keywords:
                if volume["fields"][key] == keywords[key]:
                    volumes.append(volume["id"])
                    break                   
        else:            
            match = True
            for key in keywords:
                if volume["fields"][key] != keywords[key]:
                    match = False
            if match:
                volumes.append(volume["id"])

    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, f"{bucket_name}-data.json", f"{bucket_name}-data.json")

    with open(f"{bucket_name}-data.json", "r") as f:
        log = json.load(f)

    os.unlink(f"{bucket_name}-data.json")

    examples = []
    for image in log["images"]:
        if (image["id"].split("-")[0] in volumes) and ((color is None) or (image["color"] == color)):
            examples.append({"url": f"https://{bucket_name}.s3.amazonaws.com/{image['id']}.jpg", "text": image["text"]})

    if len(examples) > max_shots:
        examples = sample(examples, max_shots)
    
    return examples    

def generate_training_data(training_data_path, keywords=None, match_mode="or", max_shots=1000):
    """Generates training data for text normalization or content extraction.

    Args:
        training_data_path (str): path to a json file containing manually constructed examples
        of content extraction that will be used to train the llm

        keywords (list, optional): list of keywords that define a subset of training
        data to use in conjunction with the next parameter; if not included, all available
        training data will be used

        match_mode (str, optional): either `and` or `or`, defines subset of training data to use
        in conjunction with previous parameter

        max_shots (int, optional): defines maximum number of examples to include in conversation
        history supplied to llm

    Returns:
        List containing training data as specified by arguments supplied. 
    """    
    examples = []

    with open(training_data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    if keywords == None:
        examples = training_data["examples"]
    elif match_mode == "or":
        for example in training_data["examples"]:
            for key in keywords:
                if example[key] == keywords[key]:
                    examples.append(example)
                    break                    
    else:
        for example in training_data["examples"]:
            match = True
            for key in keywords:
                if example[key] != keywords[key]:
                    match = False
            if match:
                examples.append(example)
    
    if len(examples) > max_shots:
        examples = sample(examples, max_shots)
    
    return examples

def parse_volume_record(volume_record_path):
    """Extracts basic volume metadata from volume record.

    Args:
        volume_record_path: either path to a json file containing
        a volume record or a dictionary containing this data

    Returns:
        Augmented volume record as json and dict containing basic volume metadata. 
    """
    if type(volume_record_path) is str:
        with open(volume_record_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = volume_record_path

    if data["country"] in ["Colombia", "Cuba", "Mexico", "United States"]:
        language = "Spanish"
    else:
        language = "Portuguese"

    volume_metadata = {"type": data["type"], "language": language, "institution": data["institution"], "id": data["id"], "state": data["state"]}

    return data, volume_metadata

def parse_record_type(volume_metadata):
    """Converts volume type as recorded in metadata to natural language equivalent.

    Args:
        volume_metadata (dict): dict containing basic volume metadata

    Returns:
        String representing natural language equivalent of volume type for use in prompting. 
    """
    if volume_metadata["type"] == "baptism":
        record_type = "baptismal register"
    elif volume_metadata["type"] == "marriage":
        record_type = "marriage register"
    elif volume_metadata["type"] == "burial":
        record_type = "burial register"
    else:
        record_type = "sacramental record"

    return record_type

def collect_instructions(instructions_path, volume_metadata, mode):
    """Collects pertinent natural language instructions for model.

    Args:
        instructions_path (str): path to json file containing model instructions

        volume_metadata (dict): dict containing basic volume metadata

        mode (str): pipeline component currently being invoked, either
        `transcription`, `normalization`, or `extraction`

    Returns:
        List containing instructions to be passed to model as system messages. 
    """
    with open(instructions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if mode == "transcription":
        if volume_metadata["fields"]["country"] == "Brazil":
            language = "Portuguese"
        else:
            language = "Spanish"        

        keywords = [mode, language, volume_metadata["type"], volume_metadata["fields"]["state"]]
    else:
        keywords = [mode, volume_metadata["language"], volume_metadata["type"]]
    
    instructions = []

    #recursively checks instructions for those that match mode, language, and record type
    for instruction in data["instructions"]:
        match = True
        for keyword in instruction["cases"]:
            if keyword not in keywords:
                match = False
        if match:
            instructions.append(instruction)

    #sorts language by intended sequence as defined in source file (lower first)
    return sorted(instructions, key=lambda x: x["sequence"])

def parse_date(date):
    """Transforms ISO 8601 string date to list of integers.

    Args:
        date (str): date or date range formatted to ISO 8601 standards

    Returns:
        List of one to six integers representing the same date or date range. 
    """
    if "/" in date:
        dates = date.split("/")
        start = dates[0].split("-")
        end = dates[1].split("-")
        parts = []
        for part in start:
            parts.append(int(part))
        for part in end:
            parts.append(int(parts))        
    else:
        parts = date.split("-")
        for part in parts:
            part = int(part)
    
    return parts

def compare_dates(x, y):
    """Determines which of two dates came first.

    Args:
        x (list): list of three integers representing a date (year, month, day)

        y (list): list of three integers representing another date (year, month, day)

    Returns:
        True if the first date occurred before the second or both dates are the same, false otherwise. 
    """
    if x[0] < y[0]:
        return True
    elif x[0] > y[0]:
        return False
    else:
        if x[1] < y[1]:
            return True
        elif x[1] > y[1]:
            return False
        else:
            if x[2] < y[2]:
                return True
            elif x[2] > y[2]:
                return False
            else:
                return True
            
def complete_date(date, mode="m"):
    """Completes incomplete dates without inference.

    Args:
        date: either a date formatted to ISO 8601 standards or a list of one or two
        integers representing an incomplete date

        mode (str): complete mode, either `m` to represent a single incomplete date,
        `s` to represent an incomplete date at the start of a date range, or `e` to
        represent an incomplete date at the end of a date range

    Returns:
        A complete date (mode `s` or `e`) or date range (mode `m`) that uses all available
        information without making any assumptions. 
    """
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if type(date) == str:
        date = parse_date(date)

    if mode == "s":
        if len(date) == 1:
            return date[0], 1, 1
        else:
            return date[0], date[1], 1
    elif mode == "e":
        if len(date) == 1:
            return date[0], 12, 31
        else:
            return date[0], date[1], months[date[1] - 1]
    else:
        if len(date) == 1:
            return date[0], 1, 1, date[0], 12, 31
        else:
            return date[0], date[1], 1, date[0], date[1], months[date[1] - 1]
        
def disambiguate_people(x, y):
    """Determines whether two records refer to the same person.

    This is currently done manually, but manual disambiguations
    will eventually be used to train an automated solution.

    Args:
        x (dict): data representing all known information about a person

        y (dict): data representing all known information about another person

    Returns:
        True if these records refer to the same person, False otherwise. 
    """
    #for key in ["rank", "origin", "ethnicity", "age", "legitimate", "occupation", "phenotype", "free"]:
        #if (key in x) and (key in y) and (x[key] != y[key]):
            #return False    
    
    people = {"people": [x, y]}

    with open("temp.json", "w", encoding="utf-8") as f:
        json.dump(people, f)    

    match = input("Should these records be combined? (y/n)")    

    if match == "y":
        match = True
    else:
        match = False

    people["match"] = match    

    with open("disambiguate.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data["manual"].append(people)    

    with open("disambiguate.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    return match

# this needs to be improved to account for nuance/change over time (e.g. DOB instead of age)
def merge_records(x, y):
    """Merges the records of two people.

    Args:
        x (dict): data representing information about a person

        y (dict): different data representing information about the same person

    Returns:
        A single dict containing all information from both input dictionaries. 
    """
    for key in ["rank", "origin", "ethnicity", "age", "legitimate", "occupation", "phenotype", "free", "mentions"]:
        if (key in y) and (key not in x):
            x[key] = y[key]
        elif key in y:
            if type(x[key]) == list:
                if y[key] not in x[key]:
                    x[key].append(y[key])
            else:
                if x[key] != y[key]:
                    x[key] = [x[key], y[key]]                

    if ("titles" in x) and ("titles" in y):
        for title in y["titles"]:
            if title not in x["titles"]:
                x["titles"].append(title)
    elif "titles" in y:
        x["titles"] = y["titles"]

    if ("relationships" in x) and ("relationships" in y):
        for rel in y["relationships"]:            
            x["relationships"].append(rel)
    elif "relationships" in y:
        x["relationships"] = y["relationships"]

    if (type(x["id"]) == str) and (type(y["id"]) == str):
        x["id"] = [x["id"], y["id"]]
    elif type(x["id"]) == str:
        x["id"] = [x["id"]]
        for id in y["id"]:
            x["id"].append(id)
    elif type(y["id"]) == str:
        x["id"].append(y["id"])
    else:
        for id in y["id"]:
            x["id"].append(id)

    return x

def transform_recogito_output(json_dir):
    import json

    with open(json_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for index, image in enumerate(data['images']):
        for dex, box in enumerate(image['coords']):            
            data['images'][index]['coords'][dex] = [box[0], box[1],
                                    box[0] + box[2],
                                    box[1] + box[3]]

    with open(json_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    