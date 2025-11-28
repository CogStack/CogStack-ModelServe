import json
import socket
import random
import struct
import inspect
import os
import sys
import copy
import functools
import warnings
import requests
import time
import torch
import tarfile
import zipfile
import numpy as np
import pandas as pd
from packaging.markers import Marker
from pydantic import BaseModel
from spacy.lang.en import English
from spacy.util import filter_spans
from safetensors.torch import load_file
from transformers import PreTrainedModel, PreTrainedTokenizer
from urllib.parse import ParseResult
from functools import lru_cache
from typing import List, Optional, Dict, Callable, Any, Union, Type, TypeVar
from app.config import Settings
from app.domain import Annotation, Entity, CodeType, ModelType, Device, PromptMessage, PromptRole
from app.exception import ManagedModelException
from app.processors.prompt_factory import PromptFactory


@lru_cache
def get_settings() -> Settings:
    """
    Gets an instance of configuration settings.

    Returns:
        Settings: An instance of the configuration settings.
    """

    settings = Settings()
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if settings.SYSTEM_METRICS_LOGGING_INTERVAL_SECONDS > 0:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(settings.SYSTEM_METRICS_LOGGING_INTERVAL_SECONDS)
    else:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    return settings


def get_code_base_uri(model_name: str) -> Optional[str]:
    """
    Gets the code base URI for a given model name.

    Args:
        model_name (str): The name of the model from which the base URI is to be inferred.

    Returns:
        Optional[str]: The code base URI associated with the model name, or None if no match is found.
    """

    code_base_uris = {
        CodeType.SNOMED.value: "http://snomed.info/id",
        CodeType.ICD10.value: "https://icdcodelookup.com/icd-10/codes",
        CodeType.OPCS4.value: "https://nhsengland.kahootz.com/t_c_home/view?objectID=14270896",
        CodeType.UMLS.value: "https://uts.nlm.nih.gov/uts/umls/concept",
    }
    for code_name, base_uri in code_base_uris.items():
        if code_name.lower() in model_name.lower():
            return base_uri
    return None


def annotations_to_entities(annotations: List[Annotation], model_name: str) -> List[Entity]:
    """
    Converts a list of annotations into a list of entities.

    Args:
        annotations (List[Annotation]): A list of annotations to be converted.
        model_name (str): The name of the model to determine the knowledge base URI.

    Returns:
        List[Entity]: A list of entities created from the annotations provided.
    """

    entities = []
    code_base_uri = get_code_base_uri(model_name)
    for _, annotation in enumerate(annotations):
        entities.append(
            Entity(
                start=annotation.start,
                end=annotation.end,
                label=annotation.label_name,
                kb_id=annotation.label_id,
                kb_url=f"{code_base_uri}/{annotation.label_id}" if code_base_uri is not None else "#",
            )
        )
    return entities


def send_gelf_message(message: str, gelf_input_uri: ParseResult) -> None:
    """
    Sends a GELF formatted message to a specified GELF input URI.

    Args:
        message (str): The message to be sent.
        gelf_input_uri (ParseResult): The URI where the GELF message should be sent.
    """

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
    """
    Gets the default parameters of a function as a dictionary.

    Args:
        func (Callable): The function whose default parameters are to be extracted.

    Returns:
        Dict: A dictionary of parameter names and their default values.
    """

    signature = inspect.signature(func)
    params = {name: param.default for name, param in signature.parameters.items() if param.default is not inspect.Parameter.empty}
    return params


def json_normalize_trainer_export(trainer_export: Dict) -> pd.DataFrame:
    """
    Normalises a trainer export into a pandas DataFrame.

    Args:
        trainer_export (Dict): A dictionary containing the trainer export data.

    Returns:
        pd.DataFrame: A DataFrame with normalised data from the trainer export.
    """

    return pd.json_normalize(
        trainer_export,
        record_path=["projects", "documents", "annotations"],
        meta=[
            ["projects", "name"], ["projects", "id"], ["projects", "cuis"], ["projects", "tuis"],
            ["projects", "documents", "id"], ["projects", "documents", "name"],
            ["projects", "documents", "text"], ["projects", "documents", "last_modified"],
        ],
        sep=".",
    )


def json_normalize_medcat_entities(medcat_entities: Dict) -> pd.DataFrame:
    """
    Normalises a dictionary of entities generated by MedCAT into a pandas DataFrame.

    Args:
        medcat_entities (Dict): A dictionary containing entities returned by MedCAT.

    Returns:
        pd.DataFrame: A DataFrame with normalised data from the MedCAT entities.
    """

    result = pd.DataFrame()
    for ent in medcat_entities.get("entities", {}).values():
        ent_df = pd.json_normalize(ent)
        result = pd.concat([result, ent_df], ignore_index=True)
    return result


def json_denormalize(df: pd.DataFrame, sep: str = ".") -> List[Dict]:
    """
    Denormalises a pandas DataFrame into a list of dictionaries.

    Args:
        df (pd.DataFrame): The DataFrame to be denormalised.
        sep (str): The separator used in the DataFrame column names. Defaults to ".".

    Returns:
        List[Dict]: A list of dictionaries representing the denormalised data.
    """

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


def filter_by_concept_ids(
    trainer_export: Dict[str, Any],
    model_type: Optional[ModelType] = None,
    extra_excluded: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Filters annotations in the trainer export based on concept IDs and model type provided.

    Args:
        trainer_export (Dict[str, Any]): A dictionary containing the trainer export data.
        model_type (Optional[ModelType]): The type of model to apply ad hoc filtering. Defaults to None.
        extra_excluded (Optional[List[str]]): Additional concept IDs to exclude. Defaults to None.

    Returns:
        Dict[str, Any]: The filtered trainer export data.
    """

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
    """
    Replaces spans associated with a concept in the trainer export using a provided transform function.

    Args:
        trainer_export (Dict[str, Any]): A dictionary containing the trainer export data.
        concept_id (str): The concept ID to identify the spans to be replaced.
        transform (Callable): A function to transform the identified spans.

    Returns:
        Dict[str, Any]: The trainer export data with transformed spans.
    """

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


def breakdown_annotations(
    trainer_export: Dict[str, Any],
    target_concept_ids: List[str],
    primary_delimiter: str,
    secondary_delimiter: Optional[str] = None,
    *,
    include_delimiter: bool = True,
) -> Dict[str, Any]:
    """
    Breaks down annotations in the trainer export based on specified delimiters.

    Args:
        trainer_export (Dict[str, Any]): A dictionary containing the trainer export data.
        target_concept_ids (List[str]): A list of concept IDs to target for breakdown.
        primary_delimiter (str): The primary delimiter to split annotations.
        secondary_delimiter (Optional[str]): The secondary delimiter for further splitting. Defaults to None.
        include_delimiter (bool): Whether to include delimiters in the split annotations. Defaults to True.

    Returns:
        Dict[str, Any]: The trainer export data with broken down annotations.
    """

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


def augment_annotations(
    trainer_export: Dict,
    cui_regexes_lists: Dict[str, List[List]],
    *,
    case_sensitive: bool = True,
) -> Dict:
    """
    Augments annotations in the trainer export based on the provided concept regex patterns.

    Args:
        trainer_export (Dict): A dictionary containing the trainer export data.
        cui_regexes_lists (Dict[str, List[List]]): A dictionary where keys are concept IDs and values are lists of regex patterns.
        case_sensitive (bool): Whether the regex matching should be case sensitive. Defaults to True.

    Returns:
        Dict: The trainer export with augmented annotations.
    """

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


def safetensors_to_pytorch(
    safetensors_file_path: Union[str, os.PathLike],
    pytorch_file_path: Union[str, os.PathLike],
) -> None:
    """
    Converts a safetensors file to a PyTorch file.

    Args:
        safetensors_file_path (Union[str, os.PathLike]): The path to the input safetensors file.
        pytorch_file_path (Union[str, os.PathLike]): The path where the output PyTorch file will be saved.
    """

    state_dict = load_file(safetensors_file_path)
    torch.save(state_dict, pytorch_file_path)


def func_deprecated(message: Optional[str] = None) -> Callable:
    """
    Decorator to mark a function as deprecated.

    Args:
        message (Optional[str]): An optional message to include in the deprecation warning.

    Returns:
        Callable: The decorated function.
    """

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
    """
    Decorator to mark a class as deprecated.

    Args:
        message (Optional[str]): An optional message to include in the deprecation warning.

    Returns:
        Callable: The decorated class.
    """

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


def reset_random_seed(seed: int = 42) -> None:
    """
    Resets the random seed for libraries used in CMS.

    Args:
        seed (int): The seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def non_default_device_is_available(device: str) -> bool:
    """
    Checks if the specified non-default device is available.

    Args:
        device (str): The string representation of the device.

    Returns:
        bool: True if the device is available, False otherwise.
    """

    return any([
        device.startswith(Device.GPU.value) and torch.cuda.is_available(),
        device.startswith(Device.MPS.value) and torch.backends.mps.is_available(),
        device.startswith(Device.CPU.value),
    ])


def get_hf_pipeline_device_id(device: str) -> int:
    """
    Retrieves the device ID for a Hugging Face pipeline based on the specified device string.

    Args:
        device (str): The string representation of the device.

    Returns:
        int: The device ID for the Hugging Face pipeline.
    """

    if device.startswith(Device.GPU.value) or device.startswith(Device.MPS.value):
        device_id = 0 if len(device.split(":")) == 1 else int(device.split(":")[1])
    else:
        device_id = -1
    return device_id


def get_hf_device_map(device: str) -> Dict:
    """
    Retrieves the device map for a Hugging Face model based on the specified device string.

    Args:
        device (str): The string representation of the device.

    Returns:
        Dict: The device map for the Hugging Face model.
    """

    if device.startswith(Device.GPU.value) or device.startswith(Device.MPS.value):
        return {"": device}
    else:
        return {"": "cpu"}


def unpack_model_data_package(model_data_file_path: str, model_data_folder_path: str) -> bool:
    """
    Unpacks a model data package from a zip or tar.gz file into the specified folder.

    Args:
        model_data_file_path (str): The path to the model data package file.
        model_data_folder_path (str): The path to the folder where the model data will be extracted.

    Returns:
        bool: True if the file was successfully unpacked, False otherwise.
    """

    if model_data_file_path.endswith(".zip"):
        with zipfile.ZipFile(model_data_file_path, "r") as f:
            f.extractall(model_data_folder_path)
            return True
    elif model_data_file_path.endswith(".tar.gz"):
        with tarfile.open(model_data_file_path, "r:gz") as f:
            for member in f.getmembers():
                f.extract(member, path=model_data_folder_path)
            return True
    else:
        return False

def get_model_data_package_base_name(file_path: str) -> str:
    """
    Gets the base name of a model data package file path.

    Args:
        file_path (str): The path to the model data package file.

    Returns:
        str: The base name of the model data package file.
    """

    if file_path.endswith(".tar.gz"):
        return os.path.basename(file_path)[:-7]
    elif file_path.endswith(".zip"):
        return os.path.basename(file_path)[:-4]
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


def create_model_data_package(model_data_folder_path: str, model_data_file_path: str) -> bool:
    """
    Creates a model data package by compressing the specified folder into a zip or tar.gz file.

    Args:
        model_data_folder_path (str): The absolute path to the folder containing the model data.
        model_data_file_path (str): The absolute path where the compressed model data package will be saved.

    Returns:
        bool: True if the package was successfully created, False otherwise.
    """

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
    """
    Gets the extension of a model data package file path.

    Args:
        file_path (str): The path to the model data package file.
        default_ext (str): The default extension to return if no extension is found. Defaults to an empty string.

    Returns:
        str: The extension of the model data package file.
    """

    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == "":
        return default_ext
    else:
        default_ext = file_ext + default_ext
        return get_model_data_package_extension(file_name, default_ext)


def ensure_tensor_contiguity(model: PreTrainedModel) -> None:
    """
    Ensures that the tensors in the Hugging Face model are contiguous.

    Args:
        model (PreTrainedModel): The model to ensure the tensors are contiguous.
    """

    for param in model.parameters():
        param.data = param.data.contiguous()


def pyproject_dependencies_to_pip_requirements(pyproject_dependencies: List[str]) -> List[str]:
    """
    Converts a list of pyproject dependencies to a list of pip requirements.

    Args:
        pyproject_dependencies (List[str]): The list of pyproject dependencies.

    Returns:
        List[str]: The list of pip requirements.
    """

    pip_requirements = []
    current_py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for dependency in pyproject_dependencies:
        if ";" in dependency:
            package, py_ver = dependency.split(";", 1)
            if Marker(py_ver.strip()).evaluate({"python_version": current_py_ver}):
                pip_requirements.append(package.strip())
        else:
            pip_requirements.append(dependency.strip())

    return pip_requirements

T = TypeVar("T", bound=BaseModel)

def load_pydantic_object_from_dict(model: Type[T], obj: Dict) -> T:
    """
    Loads the dictionary into the pydantic model passed in.

    Args:
         model (Type[T]): The pydantic model to parse the object into.
         obj (Dict): The dictionary object to load.

    Returns:
        T: The pydantic model object.
    """

    if hasattr(model, "parse_obj"):
        return model.parse_obj(obj)
    elif hasattr(model, "model_validate"):
        return model.model_validate(obj)
    else:
        raise TypeError("Model must have a known method for parsing objects.")


def dump_pydantic_object_to_dict(model: BaseModel) -> Dict:
    """
    Dumps the pydantic model object to a dictionary.

    Args:
        model (BaseModel): The pydantic model to dump.

    Returns:
        Dict: The dictionary object.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    elif hasattr(model, "dict"):
        return model.dict()
    else:
        raise TypeError("Model must have a known method for dumping objects.")

def download_model_package(
    model_package_url: str,
    destination_path: str,
    max_retries: int = 5,
    initial_delay_secs: int = 1,
    overwrite: bool = True,
) -> None:
    """
    Downloads a model package from a URL and returns the path to the downloaded file.

    Args:
        model_package_url (str): The URL of the model package to download.
        destination_path (str): The path where the downloaded model package file will be saved.
        max_retries (int): The maximum number of retries to attempt. Defaults to 5.
        initial_delay_secs (int): The initial delay in seconds between retries. Defaults to 1.
        overwrite (bool): Whether to overwrite the destination file if it already exists. Defaults to True.

    Raises:
        ManagedModelException: If the model package was not successfully downloaded after the maximum number of retries.
    """

    if os.path.exists(destination_path) and not overwrite:
        return

    retry_delay = float(initial_delay_secs)
    for attempt in range(max_retries):
        try:
            with requests.get(model_package_url, stream=True) as response:
                response.raise_for_status()
                with open(destination_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise ManagedModelException(f"Failed to download model from {model_package_url} after {max_retries} attempts: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2


def get_default_chat_template() -> str:
    """
    Gets the default chat template.

    Returns:
        str: The default chat template.
    """

    return (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if loop.index0 == 0 and system_message != false %}"
        "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
        "{% else %}"
        "{% set content = message['content'] %}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<s>[INST] ' + content + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' ' + content + ' </s>' }}"
        "{% endif %}"
        "{% endfor %}"
    )


def get_default_system_prompt() -> str:
    """
    Gets the default system prompt.

    Returns:
        str: The default system prompt.
    """
    return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., "
        "<reasoning> reasoning process here </reasoning><answer> answer here </answer>"
    )


def get_prompt_from_messages(
        tokenizer: PreTrainedTokenizer,
        messages: List[PromptMessage],
        override_template: Optional[str] = None,
) -> str:
    """
    Generates a prompt from a list of prompt messages.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for applying the chat template.
        messages (List[PromptMessage]): The list of prompt messages to use for generating the prompt.
        override_template (str): The name of the chat template to use for generating the prompt.

    Returns:
        str: The generated prompt.
    """
    if override_template is None:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [dump_pydantic_object_to_dict(message) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif hasattr(tokenizer, "default_chat_template") and tokenizer.default_chat_template:
            # This largely depends on how older versions of HF tokenizers behave and may not work universally
            tokenizer.chat_template = tokenizer.default_chat_template
            prompt = tokenizer.apply_chat_template(
                [dump_pydantic_object_to_dict(message) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            system_content = ""
            prompt_parts: List[str] = []
            for message in messages:
                content = message.content.strip()
                if message.role == PromptRole.SYSTEM:
                    system_content = content
                elif message.role == PromptRole.USER:
                    prompt_parts.append(f"<|user|>\n{content}</s>")
                elif message.role == PromptRole.ASSISTANT:
                    prompt_parts.append(f"<|assistant|>\n{content}</s>")
            if system_content:
                prompt = f"<|system|>\n{system_content}</s>\n" + "\n".join(prompt_parts)
            else:
                prompt = "\n".join(prompt_parts)
            prompt += "\n<|assistant|>\n"
    else:
        tokenizer.chat_template = PromptFactory.create_chat_template(name=override_template)
        prompt = tokenizer.apply_chat_template(
            [dump_pydantic_object_to_dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


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
