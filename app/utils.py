import json
import socket
import random
import struct
import inspect
import os
import copy
import functools
import warnings
import torch
import tarfile
import zipfile
import numpy as np
import pandas as pd
from spacy.lang.en import English
from spacy.util import filter_spans
from safetensors.torch import load_file
from urllib.parse import ParseResult
from functools import lru_cache
from typing import List, Optional, Dict, Callable, Any, Union, Type
from app.domain import Annotation, Entity, CodeType, ModelType, Device
from app.config import Settings


@lru_cache
def get_settings() -> Settings:
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
        entities.append(Entity(
            start=annotation.start,
            end=annotation.end,
            label=annotation.label_name,
            kb_id=annotation.label_id,
            kb_url=f"{code_base_uri}/{annotation.label_id}" if code_base_uri is not None else "#"
        ))
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
    sock.sendall(b"\x1e\x0f" + message_id + b"\x00\x00" + bytes(json.dumps(message), "utf-8"))
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
    for ent in medcat_entities.get("entities", {}).values():
        ent_df = pd.json_normalize(ent)
        result = pd.concat([result, ent_df], ignore_index=True)
    return result


def json_denormalize(df: pd.DataFrame, sep: str = ".") -> List[Dict]:
    result: List[Dict] = []
    for _, row in df.iterrows():
        result_row: Dict = {}
        for col, cell in row.items():
            keys = col.split(sep)
            current = result_row
            for i, k in enumerate(keys):
                if i == len(keys)-1:
                    current[k] = cell
                else:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
        result.append(result_row)
    return result


def filter_by_concept_ids(trainer_export: Dict[str, Any],
                          model_type: Optional[ModelType] = None,
                          extra_excluded: Optional[List[str]] = None) -> Dict[str, Any]:
    concept_ids = get_settings().TRAINING_CONCEPT_ID_WHITELIST.split(",")
    filtered = copy.deepcopy(trainer_export)
    for project in filtered.get("projects", []):
        for document in project.get("documents", []):
            if concept_ids == [""]:
                document["annotations"] = [anno for anno in document.get("annotations", []) if anno.get("correct", True) and not anno.get("deleted", False) and not anno.get("killed", False)]
            else:
                document["annotations"] = [anno for anno in document.get("annotations", []) if anno.get("cui") in concept_ids and anno.get("correct", True) and not anno.get("deleted", False) and not anno.get("killed", False)]

            if extra_excluded is not None and len(extra_excluded) > 0:
                document["annotations"] = [anno for anno in document.get("annotations", []) if anno.get("cui") not in extra_excluded]

    if model_type in [ModelType.TRANSFORMERS_DEID, ModelType.MEDCAT_DEID, ModelType.ANONCAT]:
        # special preprocessing for the DeID annotations and consider removing this.
        for project in filtered["projects"]:
            for document in project["documents"]:
                for annotation in document["annotations"]:
                    if annotation["cui"] == "N1100" or annotation["cui"] == "N1200":    # for metric calculation
                        annotation["cui"] = "N1000"
                    if annotation["cui"] == "W5000" and (model_type in [ModelType.MEDCAT_DEID, ModelType.ANONCAT]):    # for compatibility
                        annotation["cui"] = "C2500"

    return filtered


def replace_spans_of_concept(trainer_export: Dict[str, Any], concept_id: str, transform: Callable) -> Dict[str, Any]:
    doc_with_initials_ids = set()
    copied = copy.deepcopy(trainer_export)
    for project in copied.get("projects", []):
        for document in project.get("documents", []):
            text = document.get("text", "")
            offset = 0
            document["annotations"] = sorted(document.get("annotations", []), key=lambda annotation: annotation["start"])
            for annotation in document.get("annotations", []):
                annotation["start"] += offset
                annotation["end"] += offset
                if annotation["cui"] == concept_id and annotation.get("correct", True) and not annotation.get("deleted", False) and not annotation.get("killed", False):
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
                          *,
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


def augment_annotations(trainer_export: Dict, cui_regexes_lists: Dict[str, List[List]], *, case_sensitive: bool = True) -> Dict:
    nlp = English()
    patterns = []
    for cui, regexes in cui_regexes_lists.items():
        pts = [{
            "label": cui,
            "pattern": [{"TEXT": {"REGEX": part if case_sensitive else r"(?i)" + part}} for part in regex]
        } for regex in regexes]
        patterns += pts
    ruler = nlp.add_pipe("entity_ruler")
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
            spans = filter_spans(doc.ents)
            for span in spans:
                for gap in gaps:
                    if span.start_char >= gap[0] and span.end_char <= gap[1]:
                        annotation = {
                            "cui": span.label_,
                            "value": span.text,
                            "start": span.start_char,
                            "end": span.end_char,
                            "correct": True,
                            "killed": False,
                            "deleted": False,
                            "manually_created": False,
                        }
                        new_annotations.append(annotation)
                        break
            document["annotations"] += new_annotations
            document["annotations"] = sorted(document["annotations"], key=lambda anno: anno["start"])

    return copied


def safetensors_to_pytorch(safetensors_file_path: Union[str, os.PathLike],
                           pytorch_file_path: Union[str, os.PathLike]) -> None:
    state_dict = load_file(safetensors_file_path)
    torch.save(state_dict, pytorch_file_path)


def func_deprecated(message: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Callable:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("Function {} has been deprecated.{}".format(func.__name__, " " + message if message else ""), stacklevel=2)
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)
        return wrapped
    return decorator


def cls_deprecated(message: Optional[str] = None) -> Callable:
    def decorator(cls: Type) -> Callable:
        decorated_init = cls.__init__

        @functools.wraps(decorated_init)
        def wrapped(self: "Type", *args: Any, **kwargs: Any) -> Any:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("Class {} has been deprecated.{}".format(cls.__name__, " " + message if message else ""))
            warnings.simplefilter("default", DeprecationWarning)
            decorated_init(self, *args, **kwargs)
        cls.__init__ = wrapped
        return cls
    return decorator


def reset_random_seed() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def non_default_device_is_available(device: str) -> bool:
    return any([
        device.startswith(Device.GPU.value) and torch.cuda.is_available(),
        device.startswith(Device.MPS.value) and torch.backends.mps.is_available(),
        device.startswith(Device.CPU.value)
    ])


def get_hf_pipeline_device_id(device: str) -> int:
    if device.startswith(Device.GPU.value) or device.startswith(Device.MPS.value):
        device_id = 0 if len(device.split(":")) == 1 else int(device.split(":")[1])
    else:
        device_id = -1
    return device_id


def unpack_model_data_package(model_data_file_path: str, model_data_folder_path: str) -> bool:
    if model_data_file_path.endswith(".zip"):
        with zipfile.ZipFile(model_data_file_path, "r") as f:
            f.extractall(model_data_folder_path)
            return True
    elif model_data_file_path.endswith(".tar.gz"):
        with tarfile.open(model_data_file_path, "r:gz") as f:
            for member in f.getmembers():
                path_parts = member.name.split(os.sep)
                stripped_path = os.sep.join(path_parts[1:])
                if not stripped_path:
                    continue
                member.name = stripped_path
                f.extract(member, path=model_data_folder_path)
            return True
    else:
        return False


def create_model_data_package(model_data_folder_path: str, model_data_file_path: str) -> bool:
    if model_data_file_path.endswith(".zip"):
        with zipfile.ZipFile(model_data_file_path, "w", zipfile.ZIP_DEFLATED) as f:
            for root, dirs, files in os.walk(model_data_folder_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), model_data_folder_path)
                    f.write(os.path.join(root, file), rel_path)
            return True
    elif model_data_file_path.endswith(".tar.gz"):
        with tarfile.open(model_data_file_path, "w:gz") as f:
            for root, dirs, files in os.walk(model_data_folder_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), model_data_folder_path)
                    f.add(os.path.join(root, file), rel_path)
            return True
    else:
        return False


def get_model_data_package_extension(file_path: str, default_ext: str = "") -> str:
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == "":
        return default_ext
    else:
        default_ext = file_ext + default_ext
        return get_model_data_package_extension(file_name, default_ext)


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
