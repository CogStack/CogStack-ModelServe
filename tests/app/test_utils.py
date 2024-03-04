import os
import json

from urllib.parse import urlparse
from utils import (
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
)


def test_get_code_base_uri():
    assert get_code_base_uri("SNOMED model") == "http://snomed.info/id"
    assert get_code_base_uri("ICD-10 model") == "https://icdcodelookup.com/icd-10/codes"
    assert get_code_base_uri("UMLS model") == "https://uts.nlm.nih.gov/uts/umls/concept"


def test_annotations_to_entities():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
        "end": 15,
    }]
    expected = [{
        "start": 1,
        "end": 15,
        "label": "Spinal stenosis",
        "kb_id": "76107001",
        "kb_url": "http://snomed.info/id/76107001",
    }]
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
    assert df.columns.tolist() == ["pretty_name", "cui", "type_ids", "types", "source_value", "detected_name", "acc", "context_similarity", "start", "end", "icd10", "ontologies", "snomed", "id", "meta_anns.Presence.value", "meta_anns.Presence.confidence", "meta_anns.Presence.name", "meta_anns.Subject.value", "meta_anns.Subject.confidence", "meta_anns.Subject.name", "meta_anns.Time.value", "meta_anns.Time.confidence", "meta_anns.Time.name"]


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
    config.TRAINING_CONCEPT_ID_WHITELIST = "C0017168"
    trainer_export_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(trainer_export_path, "r") as f:
        trainer_export = json.load(f)
    filtered = filter_by_concept_ids(trainer_export)
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
    result = breakdown_annotations(trainer_export, ["C0017168"], " ", "e", False)
    for project in result["projects"]:
        for document in project["documents"]:
            for annotation in document["annotations"]:
                if annotation["cui"] == "C0017168":
                    assert annotation["value"] in ["gastro", "sophag", "al", "r", "flux"]
