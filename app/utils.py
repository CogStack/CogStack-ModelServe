import json
import socket
import random
import struct
import inspect
import os
import copy
from spacy.lang.en import English
import pandas as pd

from urllib.parse import ParseResult
from functools import lru_cache
from typing import List, Optional, Dict, Callable, Any
from domain import Annotation, Entity, CodeType, ModelType
from config import Settings


@lru_cache()
def get_settings() -> Settings:
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    return Settings()


def get_code_base_uri(model_name: str) -> Optional[str]:
    code_base_uris = {
        CodeType.SNOMED.value: "http://snomed.info/id",
        CodeType.ICD10.value: "https://icdcodelookup.com/icd-10/codes",
        CodeType.UMLS.value: "https://uts.nlm.nih.gov/uts/umls/concept",
    }
    for code_name, base_uri in code_base_uris.items():
        if code_name.lower() in model_name.lower():
            return base_uri
    return None


def annotations_to_entities(annotations: List[Annotation], model_name: str) -> List[Entity]:
    entities = []
    code_base_uri = get_code_base_uri(model_name)
    for _, annotation in enumerate(annotations):
        entities.append({
            "start": annotation["start"],
            "end": annotation["end"],
            "label": f"{annotation['label_name']}",
            "kb_id": annotation["label_id"],
            "kb_url": f"{code_base_uri}/{annotation['label_id']}" if code_base_uri is not None else "#"
        })
    return entities


def send_gelf_message(message: str, gelf_input_uri: ParseResult) -> None:
    message = {
        "version": "1.1",
        "host": socket.gethostname(),
        "short_message": message,
        "level": 1,
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((gelf_input_uri.hostname, gelf_input_uri.port))

    message_id = struct.pack("<Q", random.getrandbits(64))
    sock.sendall(b'\x1e\x0f' + message_id + b'\x00\x00' + bytes(json.dumps(message), "utf-8"))
    sock.close()


def get_func_params_as_dict(func: Callable) -> Dict:
    signature = inspect.signature(func)
    params = {name: param.default for name, param in signature.parameters.items() if param.default is not inspect.Parameter.empty}
    return params


def json_normalize_trainer_export(trainer_export: Dict) -> pd.DataFrame:
    return pd.json_normalize(trainer_export,
                             record_path=["projects", "documents", "annotations"],
                             meta=[
                                    ["projects", "name"], ["projects", "id"], ["projects", "cuis"], ["projects", "tuis"],
                                    ["projects", "documents", "id"], ["projects", "documents", "name"],
                                    ["projects", "documents", "text"], ["projects", "documents", "last_modified"]
                             ],
                             sep=".")


def json_normalize_medcat_entities(medcat_entities: Dict) -> pd.DataFrame:
    result = pd.DataFrame()
    for _, ent in medcat_entities["entities"].items():
        ent_df = pd.json_normalize(ent)
        result = pd.concat([result, ent_df], ignore_index=True)
    return result


def json_denormalize(df: pd.DataFrame, sep: str = ".") -> List[Dict]:
    result: List[Dict] = []
    for idx, row in df.iterrows():
        result_row: Dict = {}
        for col, cell in row.items():
            keys = col.split(sep)
            current = result_row
            for i, k in enumerate(keys):
                if i == len(keys)-1:
                    current[k] = cell
                else:
                    if k not in current.keys():
                        current[k] = {}
                    current = current[k]
        result.append(result_row)
    return result


def filter_by_concept_ids(trainer_export: Dict[str, Any], model_type: Optional[ModelType] = None) -> Dict[str, Any]:
    concept_ids = get_settings().TRAINING_CONCEPT_ID_WHITELIST.split(",")
    filtered = copy.deepcopy(trainer_export)
    for project in filtered["projects"]:
        for document in project["documents"]:
            if concept_ids == [""]:
                document["annotations"] = [anno for anno in document["annotations"] if anno["correct"] and not anno["deleted"] and not anno["killed"]]
            else:
                document["annotations"] = [anno for anno in document["annotations"] if anno["cui"] in concept_ids and anno["correct"] and not anno["deleted"] and not anno["killed"]]

    if model_type == ModelType.TRANSFORMERS_DEID or model_type == ModelType.MEDCAT_DEID:
        # special preprocessing for the DeID annotations and consider removing this.
        for project in filtered["projects"]:
            for document in project["documents"]:
                for annotation in document["annotations"]:
                    if annotation["cui"] == "N1100" or annotation["cui"] == "N1200":    # for metric calculation
                        annotation["cui"] = "N1000"
                    if annotation["cui"] == "W5000" and model_type == ModelType.MEDCAT_DEID:    # for compatibility
                        annotation["cui"] = "C2500"

    return filtered


def replace_spans_of_concept(trainer_export: Dict[str, Any], concept_id: str, transform: Callable) -> Dict[str, Any]:
    doc_with_initials_ids = set()
    copied = copy.deepcopy(trainer_export)
    for project in copied["projects"]:
        for document in project["documents"]:
            text = document["text"]
            offset = 0
            document["annotations"] = sorted(document["annotations"], key=lambda annotation: annotation["start"])
            for annotation in document["annotations"]:
                annotation["start"] += offset
                annotation["end"] += offset
                if annotation["cui"] == concept_id and annotation["correct"] and not annotation["deleted"] and not annotation["killed"]:
                    original = annotation["value"]
                    modified = transform(original)
                    extended = len(modified) - len(original)
                    text = text[:annotation["start"]] + modified + text[annotation["end"]:]
                    annotation["value"] = modified
                    annotation["end"] += extended
                    offset += extended
                    doc_with_initials_ids.add(document["id"])
            document["text"] = text
    return copied


def breakdown_annotations(trainer_export: Dict[str, Any],
                          target_concept_ids: List[str],
                          primary_delimiter: str,
                          secondary_delimiter: Optional[str] = None,
                          include_delimiter: bool = True) -> Dict[str, Any]:
    assert isinstance(target_concept_ids, list), "The target_concept_ids is not a list"
    copied = copy.deepcopy(trainer_export)
    for project in copied["projects"]:
        for document in project["documents"]:
            new_annotations = []
            for annotation in document["annotations"]:
                if annotation["cui"] in target_concept_ids and primary_delimiter in annotation["value"]:
                    start_offset = 0
                    for sub_text in annotation["value"].split(primary_delimiter):
                        if secondary_delimiter is not None and secondary_delimiter in sub_text:
                            for sub_sub_text in sub_text.split(secondary_delimiter):
                                if sub_sub_text == "" or all(char.isspace() for char in sub_sub_text):
                                    start_offset += len(sub_sub_text) + len(secondary_delimiter)
                                    continue
                                sub_sub_annotation = copy.deepcopy(annotation)
                                sub_sub_annotation["start"] = annotation["start"] + start_offset
                                sub_sub_annotation["end"] = sub_sub_annotation["start"] + len(sub_sub_text) + (len(secondary_delimiter) if include_delimiter else 0)
                                sub_sub_annotation["value"] = sub_sub_text + (secondary_delimiter if include_delimiter else "")
                                start_offset += len(sub_sub_text) + len(secondary_delimiter)
                                new_annotations.append(sub_sub_annotation)
                            if include_delimiter:
                                new_annotations[-1]["value"] = new_annotations[-1]["value"][:-len(secondary_delimiter)] + primary_delimiter
                        else:
                            if sub_text == "" or all(char.isspace() for char in sub_text):
                                start_offset += len(sub_text) + len(primary_delimiter)
                                continue
                            sub_annotation = copy.deepcopy(annotation)
                            sub_annotation["start"] = annotation["start"] + start_offset
                            sub_annotation["end"] = sub_annotation["start"] + len(sub_text) + (len(primary_delimiter) if include_delimiter else 0)
                            sub_annotation["value"] = sub_text + (primary_delimiter if include_delimiter else "")
                            start_offset += len(sub_text) + len(primary_delimiter)
                            new_annotations.append(sub_annotation)
                    if include_delimiter:
                        new_annotations[-1]["end"] -= len(primary_delimiter)
                        new_annotations[-1]["value"] = new_annotations[-1]["value"][:-len(primary_delimiter)]
                else:
                    new_annotations.append(annotation)
            document["annotations"] = new_annotations
    return copied


def augment_annotations(trainer_export: Dict, cuis: List, regex_lists: List[List]) -> Dict:
    nlp = English()
    patterns = []
    for cui, regexes in zip(cuis, regex_lists):
        pattern = {
            "label": cui,
            "pattern": [{"TEXT": {"REGEX": regex} for regex in regexes}],
        }
        patterns.append(pattern)
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    ruler.add_patterns(patterns)    # type: ignore
    copied = copy.deepcopy(trainer_export)
    for project in copied["projects"]:
        for document in project["documents"]:
            document["annotations"] = sorted(document["annotations"], key=lambda anno: anno["start"])
            gaps = []
            gap_start = 0
            for annotation in document["annotations"]:
                if gap_start < annotation["start"]:
                    gaps.append((gap_start, annotation["start"]))
                gap_start = annotation["end"]
            if gap_start < len(document["text"]):
                gaps.append((gap_start, len(document["text"])+1))
            new_annotations = []
            doc = nlp(document["text"])
            for ent in doc.ents:
                start = ent.start_char
                end = ent.end_char
                for gap in gaps:
                    if start >= gap[0] and end <= gap[1]:
                        annotation = {
                            "cui": ent.label_,
                            "value": ent.text,
                            "start": start,
                            "end": end,
                            "correct": True,
                            "killed": False,
                            "manually_created": False,
                        }
                        new_annotations.append(annotation)
                        break
            document["annotations"] += new_annotations
            document["annotations"] = sorted(document["annotations"], key=lambda anno: anno["start"])

    return copied


TYPE_ID_TO_NAME_PATCH = {
    "32816260": "physical object",
    "2680757": "observable entity",
    "37552161": "body structure",
    "91776366": "product",
    "81102976": "organism",
    "28321150": "procedure",
    "67667581": "finding",
    "7882689": "qualifier value",
    "91187746": "substance",
    "29422548": "core metadata concept",
    "40357424": "foundation metadata concept",
    "33782986": "morphologic abnormality",
    "9090192": "disorder",
    "90170645": "record artifact",
    "66527446": "body structure",
    "3061879": "situation",
    "16939031": "occupation",
    "31601201": "person",
    "37785117": "medicinal product",
    "17030977": "assessment scale",
    "47503797": "regime/therapy",
    "33797723": "event",
    "82417248": "navigational concept",
    "75168589": "environment",
    "9593000": "medicinal product form",
    "99220404": "cell",
    "13371933": "social concept",
    "46922199": "religion/philosophy",
    "20410104": "ethnic group",
    "27603525": "clinical drug",
    "43039974": "qualifier value",
    "43857361": "physical force",
    "40584095": "metadata",
    "337250": "specimen",
    "46506674": "disposition",
    "87776218": "role",
    "30703196": "tumor staging",
    "31685163": "staging scale",
    "21114934": "dose form",
    "70426313": "namespace concept",
    "51120815": "intended site",
    "45958968": "administration method",
    "51885115": "OWL metadata concept",
    "8067332": "Lyophilized Dosage Form Category",
    "95475658": "product name",
    "43744943": "supplier",
    "66203715": "transformation",
    "64755083": "release characteristic",
    "49144999": "state of matter",
    "39041339": "unit of presentation",
    "18854038": "geographic location",
    "3242456": "life style",
    "28695783": "link assertion",
    "14654508": "racial group",
    "92873870": "special concept",
    "78096516": "environment / location",
    "72706784": "context-dependent category",
    "25624495": '© 2002-2020 International Health Terminology Standards Development Organisation (IHTSDO). All rights reserved. SNOMED CT®, was originally created by The College of American Pathologists. "SNOMED" and "SNOMED CT" are registered trademarks of the IHTSDO.',
    "55540447": "linkage concept"
}
