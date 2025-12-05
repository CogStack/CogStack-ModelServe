import os
import json
import tempfile
import torch
import shutil
import zipfile
import tarfile
import unittest
from unittest.mock import MagicMock, patch
from safetensors.torch import save_file
from transformers import PreTrainedModel
from urllib.parse import urlparse
from app.utils import (
    get_settings,
    get_code_base_uri,
    annotations_to_entities,
    send_gelf_message,
    get_func_params_as_dict,
    json_normalize_medcat_entities,
    json_normalize_trainer_export,
    json_denormalize,
    filter_by_concept_ids,
    replace_spans_of_concept,
    breakdown_annotations,
    augment_annotations,
    safetensors_to_pytorch,
    non_default_device_is_available,
    get_hf_pipeline_device_id,
    get_hf_device_map,
    get_model_data_package_extension,
    unpack_model_data_package,
    create_model_data_package,
    ensure_tensor_contiguity,
    pyproject_dependencies_to_pip_requirements,
    get_model_data_package_base_name,
    load_pydantic_object_from_dict,
    dump_pydantic_object_to_dict,
    get_prompt_from_messages,
)
from app.domain import Annotation, Entity, PromptMessage, PromptRole


def test_get_code_base_uri():
    assert get_code_base_uri("SNOMED model") == "http://snomed.info/id"
    assert get_code_base_uri("ICD-10 model") == "https://icdcodelookup.com/icd-10/codes"
    assert get_code_base_uri("OPCS-4 model") == "https://nhsengland.kahootz.com/t_c_home/view?objectID=14270896"
    assert get_code_base_uri("UMLS model") == "https://uts.nlm.nih.gov/uts/umls/concept"


def test_annotations_to_entities():
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "Spinal stenosis",
                "label_id": "76107001",
                "start": 1,
                "end": 15,
            },
        ),
    ]
    expected = [
        load_pydantic_object_from_dict(
            Entity,
            {
                "start": 1,
                "end": 15,
                "label": "Spinal stenosis",
                "kb_id": "76107001",
                "kb_url": "http://snomed.info/id/76107001",
            },
        ),
    ]
    assert annotations_to_entities(annotations, "SNOMED model") == expected


def test_send_gelf_message(mocker):
    mocked_socket = mocker.Mock()
    mocked_socket_socket = mocker.Mock(return_value=mocked_socket)
    mocker.patch("socket.socket", new=mocked_socket_socket)
    send_gelf_message("message", urlparse("http://127.0.0.1:12201"))
    mocked_socket.connect.assert_called_once_with(("127.0.0.1", 12201))
    mocked_socket.sendall.assert_called_once()
    assert b"\x1e\x0f" in mocked_socket.sendall.call_args[0][0]
    assert b"\x00\x00" in mocked_socket.sendall.call_args[0][0]
    mocked_socket.close.assert_called_once()


def test_get_func_params_as_dict():
    def func(arg1, arg2=None, arg3="arg3"):
        pass
    params = get_func_params_as_dict(func)
    assert params == {"arg2": None, "arg3": "arg3"}


def test_json_normalize_medcat_entities():
    medcat_entities_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "medcat_entities.json")
    with open(medcat_entities_path, "r") as f:
        medcat_entities = json.load(f)
    df = json_normalize_medcat_entities(medcat_entities)
    assert len(df) == 25
    assert df.columns.tolist() == ["pretty_name", "cui", "type_ids", "types", "source_value", "detected_name", "acc", "context_similarity", "start", "end", "icd10", "opcs4", "ontologies", "snomed", "id", "meta_anns.Presence.value", "meta_anns.Presence.confidence", "meta_anns.Presence.name", "meta_anns.Subject.value", "meta_anns.Subject.confidence", "meta_anns.Subject.name", "meta_anns.Time.value", "meta_anns.Time.confidence", "meta_anns.Time.name"]


def test_json_normalize_trainer_export():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    df = json_normalize_trainer_export(trainer_export)
    assert len(df) == 30
    assert df.columns.tolist() == ["id", "user", "cui", "value", "start", "end", "validated", "correct", "deleted", "alternative", "killed", "last_modified", "manually_created", "acc", "meta_anns.Status.name", "meta_anns.Status.value", "meta_anns.Status.acc", "meta_anns.Status.validated", "projects.name", "projects.id", "projects.cuis", "projects.tuis", "projects.documents.id", "projects.documents.name", "projects.documents.text", "projects.documents.last_modified"]


def test_json_denormalize():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    df = json_normalize_trainer_export(trainer_export)
    trainer_export = json_denormalize(df)
    assert len(trainer_export) == 30


def test_filter_by_concept_ids():
    config = get_settings()
    backup = config.TRAINING_CONCEPT_ID_WHITELIST
    config.TRAINING_CONCEPT_ID_WHITELIST = "C0017168, C0020538"
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    filtered = filter_by_concept_ids(trainer_export, extra_excluded=["C0020538"])
    for project in filtered["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                assert annotation["cui"] == "C0017168"
    config.TRAINING_CONCEPT_ID_WHITELIST = backup


def test_replace_spans_of_concept():
    def transform(source: str) -> str:
        return source.upper()[:-7]
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    result = replace_spans_of_concept(trainer_export, "C0017168", transform)
    updated = [(anno["value"], anno["start"], anno["end"]) for anno in result["projects"][0]["documents"][0]["annotations"] if anno["cui"] == "C0017168"]
    assert updated[0][0] == "GASTROESOPHAGEAL"
    assert updated[0][1] == 332
    assert updated[0][2] == 348


def test_breakdown_annotations():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    result = breakdown_annotations(trainer_export, ["C0017168"], " ", "e")
    for project in result["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                if annotation["cui"] == "C0017168":
                    assert annotation["value"] in ["gastroe", "sophage", "al ", "re", "flux"]


def test_breakdown_annotations_without_including_delimiter():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    result = breakdown_annotations(trainer_export, ["C0017168"], " ", "e", include_delimiter=False)
    for project in result["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                if annotation["cui"] == "C0017168":
                    assert annotation["value"] in ["gastro", "sophag", "al", "r", "flux"]


def test_augment_annotations():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    result = augment_annotations(trainer_export, {"00001": [["HISTORY"]], "00002": [["DISCHARGE"]]})
    match_count_00001 = 0
    match_count_00002 = 0
    for project in result["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                if annotation["cui"] == "00001":
                    match_count_00001 += 1
                if annotation["cui"] == "00002":
                    match_count_00002 += 1
    assert match_count_00001 == 5
    assert match_count_00002 == 1


def test_augment_annotations_case_insensitive():
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    result = augment_annotations(trainer_export, {
        "00001": [["HiSToRy"]],
        "00002": [
            [r"^\d{1,2}\s*$", r"-", r"^\s*\d{1,2}\s*$", r"-", r"^\s*\d{2,4}$"],
            [r"^\d{1,2}\s*[.\/]\s*\d{1,2}\s*[.\/]\s*\d{2,4}$"],
            [r"^\d{2,4}\s*$", r"-", r"^\s*\d{1,2}\s*$", r"-", r"^\s*\d{1,2}$"],
            [r"^\d{2,4}\s*[.\/]\s*\d{1,2}\s*[.\/]\s*\d{1,2}$"],
            [r"^\d{1,2}$", r"^[-.\/]$", r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s*[-.\/]\s*\d{2,4}$"],
            [r"^\d{2,4}$", r"^[-.\/]$", r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s*[-.\/]\s*\d{1,2}$"],
            [r"^\d{1,2}\s*$", r"-", r"^\s*\d{4}$"],
            [r"^\d{1,2}\s*[\/]\s*\d{4}$"],
            [r"^\d{4}\s*$", r"-", r"^\s*\d{1,2}$"],
            [r"^\d{4}\s*[\/]\s*\d{1,2}$"],
            [r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s*[-.\/]\s*\d{4}$"],
            [r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)(\s+\d{1,2})*$", r",", r"^\d{4}$"],
            [r"^\d{4}\s*[-.\/]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)$"],
            [r"^\d{4}$", r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)$"],
            [r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)$", r"^\d{4}$"],
            [r"^(?:19\d\d|20\d\d)$"],
        ]
    }, case_sensitive=False)

    match_count_00001 = 0
    match_count_00002 = 0
    for project in result["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                if annotation["cui"] == "00001":
                    match_count_00001 += 1
                if annotation["cui"] == "00002":
                    match_count_00002 += 1
    assert match_count_00001 == 10
    assert match_count_00002 == 4


def test_safetensors_to_pytorch():
    with tempfile.NamedTemporaryFile() as input:
        model = _DummyModel()
        model(torch.randn(1, 10))
        save_file(model.state_dict(), input.name)
        input.flush()

        with tempfile.NamedTemporaryFile() as output:
            assert not bool(output.readline())
            safetensors_to_pytorch(input.name, output.name)
            assert bool(output.readline())


def test_non_default_device_is_available():
    assert not non_default_device_is_available("default")
    assert non_default_device_is_available("cpu")


def test_get_hf_pipeline_device_id():
    assert get_hf_pipeline_device_id("cpu") == -1
    assert get_hf_pipeline_device_id("cuda") == 0
    assert get_hf_pipeline_device_id("cuda:0") == 0
    assert get_hf_pipeline_device_id("mps:1") == 1


def test_get_hf_device_map():
    assert get_hf_device_map("cuda") == {"": "cuda"}
    assert get_hf_device_map("mps") == {"": "mps"}
    assert get_hf_device_map("cpu") == {"": "cpu"}
    assert get_hf_device_map("unknown") == {"": "cpu"}


def test_get_model_data_package_extension():
    assert get_model_data_package_extension("model.zip") == ".zip"
    assert get_model_data_package_extension("model.tar.gz") == ".tar.gz"
    assert get_model_data_package_extension("model") == ""
    assert get_model_data_package_extension("") == ""


def test_ensure_tensor_contiguity():
    mock_model = MagicMock(spec=PreTrainedModel)
    param1 = torch.randn(5, 5)[:, ::2]
    param2 = torch.randn(3, 6)[:, ::2]
    mock_model.parameters.return_value = [
        MagicMock(data=param1),
        MagicMock(data=param2),
    ]
    assert not param1.is_contiguous()
    assert not param2.is_contiguous()

    ensure_tensor_contiguity(mock_model)

    for param in mock_model.parameters():
        assert param.data.is_contiguous() == True


def test_pyproject_dependencies_to_pip_requirements():
    pyproject_dependencies = [
        "package~=1.2.3; python_version >= '3.10'",
        "package~=0.1.2; python_version < '3.10'",
        "another-package~=2.3.4",
    ]

    result = pyproject_dependencies_to_pip_requirements(pyproject_dependencies)

    assert len(result) == 2
    assert result[1] == "another-package~=2.3.4"


def test_get_model_data_package_base_name():
    assert get_model_data_package_base_name("/path/to/model.zip") == "model"
    assert get_model_data_package_base_name("/path/to/model.tar.gz") == "model"
    assert get_model_data_package_base_name("/path/to/model") == "model"


class TestUnpackModelPackage(unittest.TestCase):
    def setUp(self):
        self.model_folder_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.model_folder_path)

    def test_unpack_zip_model_package(self):
        model_file_path = os.path.join(self.model_folder_path, "model.zip")
        with zipfile.ZipFile(model_file_path, "w") as f:
            f.writestr("file1.txt", "content1")
            f.writestr("subdir/file2.txt", "content2")

        result = unpack_model_data_package(model_file_path, self.model_folder_path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.model_folder_path, "file1.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.model_folder_path, "subdir/file2.txt")))

    def test_unpack_tar_gz_model_package(self):
        model_file_path = os.path.join(self.model_folder_path, "model.tar.gz")
        with tarfile.open(model_file_path, "w:gz") as f:
            file1_path = os.path.join(self.model_folder_path, "file1.txt")
            with open(file1_path, "w") as f1:
                f1.write("content1")
            f.add(file1_path, arcname="top_level/file1.txt")
            subdir_path = os.path.join(self.model_folder_path, "subdir")
            os.makedirs(subdir_path)
            file2_path = os.path.join(subdir_path, "file2.txt")
            with open(file2_path, "w") as f2:
                f2.write("content2")
            f.add(file2_path, arcname="top_level/subdir/file2.txt")

        result = unpack_model_data_package(model_file_path, self.model_folder_path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.model_folder_path, "file1.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.model_folder_path, "subdir/file2.txt")))

    def test_unpack_unsupported_model_package(self):
        unknown_model_file_path = os.path.join(self.model_folder_path, "model.unknown")
        with open(unknown_model_file_path, "w") as f:
            f.write("content")

        result = unpack_model_data_package(unknown_model_file_path, self.model_folder_path)

        self.assertFalse(result)


class TestCreateModelPackage(unittest.TestCase):
    def setUp(self):
        self.model_folder_path = tempfile.mkdtemp()
        with open(os.path.join(self.model_folder_path, "file1.txt"), "w") as f:
            f.write("content1")
        subdir_path = os.path.join(self.model_folder_path, "subdir")
        os.makedirs(subdir_path)
        with open(os.path.join(subdir_path, "file2.txt"), "w") as f:
            f.write("content2")

    def tearDown(self):
        shutil.rmtree(self.model_folder_path)

    def test_create_zip_model_package(self):
        model_file_path = os.path.join(self.model_folder_path, "model.zip")

        result = create_model_data_package(self.model_folder_path, model_file_path)

        self.assertTrue(result)
        with zipfile.ZipFile(model_file_path, "r") as f:
            extracted_files = f.namelist()
            self.assertIn("file1.txt", extracted_files)
            self.assertIn("subdir/file2.txt", extracted_files)

    def test_create_tar_gz_model_package(self):
        model_file_path = os.path.join(self.model_folder_path, "model.tar.gz")

        result = create_model_data_package(self.model_folder_path, model_file_path)

        self.assertTrue(result)
        with tarfile.open(model_file_path, "r:gz") as f:
            extracted_files = [member.name for member in f.getmembers()]
            self.assertIn("file1.txt", extracted_files)
            self.assertIn("subdir/file2.txt", extracted_files)

    def test_create_unsupported_model_package(self):
        unknown_model_file_path = os.path.join(self.model_folder_path, "model.unknown")

        result = create_model_data_package(self.model_folder_path, unknown_model_file_path)

        self.assertFalse(result)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super(_DummyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_get_prompt_with_chat_template():
    with patch('transformers.PreTrainedTokenizer') as tok:
        mock_tokenizer = tok.return_value
        mock_tokenizer.chat_template = "Mock chat template"
        mock_tokenizer.apply_chat_template.return_value = "Mock chat template applied"
        messages = [
            PromptMessage(content="Alright?", role=PromptRole.USER.value),
            PromptMessage(content="Yeah.", role=PromptRole.ASSISTANT.value),
        ]

        prompt = get_prompt_from_messages(mock_tokenizer, messages)

        assert prompt == "Mock chat template applied"


def test_get_prompt_with_default_chat_template():
    with patch('transformers.PreTrainedTokenizer') as tok:
        mock_tokenizer = tok.return_value
        mock_tokenizer.chat_template = None
        mock_tokenizer.default_chat_template = "Mock default chat template"
        mock_tokenizer.apply_chat_template.return_value = "Mock default chat template applied"
        messages = [
            PromptMessage(content="Alright?", role=PromptRole.USER.value),
            PromptMessage(content="Yeah.", role=PromptRole.ASSISTANT.value),
        ]

        prompt = get_prompt_from_messages(mock_tokenizer, messages)

        assert prompt == "Mock default chat template applied"


def test_get_prompt_without_chat_template():
    with patch('transformers.PreTrainedTokenizer') as tok:
        mock_tokenizer = tok.return_value
        mock_tokenizer.chat_template = None
        mock_tokenizer.default_chat_template = None
        messages = [
            PromptMessage(content="You are a helpful assistant.", role=PromptRole.SYSTEM.value),
            PromptMessage(content="Alright?", role=PromptRole.USER.value),
            PromptMessage(content="Yeah.", role=PromptRole.ASSISTANT.value),
        ]

        prompt = get_prompt_from_messages(mock_tokenizer, messages)

        expected_prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nAlright?</s>\n<|assistant|>\nYeah.</s>\n<|assistant|>\n"
        assert prompt == expected_prompt


def test_get_prompt_with_no_messages():
    with patch('transformers.PreTrainedTokenizer') as tok:
        mock_tokenizer = tok.return_value
        mock_tokenizer.chat_template = None
        mock_tokenizer.default_chat_template = None
        messages = []

        prompt = get_prompt_from_messages(mock_tokenizer, messages)

        assert prompt == "\n<|assistant|>\n"
