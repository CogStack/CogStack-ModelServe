import os
import tempfile

import httpx
import json
import pytest
import api.globals as cms_globals
from fastapi.testclient import TestClient
from api.api import get_model_server
from utils import get_settings
from model_services.medcat_model import MedCATModel
from management.model_manager import ModelManager
from unittest.mock import create_autospec

config = get_settings()
config.ENABLE_TRAINING_APIS = "true"
config.DISABLE_UNSUPERVISED_TRAINING = "false"
config.ENABLE_EVALUATION_APIS = "true"
config.ENABLE_PREVIEWS_APIS = "true"
config.AUTH_USER_ENABLED = "true"

TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export.json")
NOTE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "note.txt")
ANOTHER_TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "another_trainer_export.json")
TRAINER_EXPORT_MULTI_PROJS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "trainer_export_multi_projs.json")
MULTI_TEXTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fixture", "sample_texts.json")


@pytest.fixture(scope="function")
def model_service():
    return create_autospec(MedCATModel)


@pytest.fixture(scope="function")
def client(model_service):
    app = get_model_server(msd_overwritten=lambda: model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    client = TestClient(app)
    yield client
    client.app.dependency_overrides.clear()


def test_process_invalid_jsonl(model_service, client):
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 0,
        "end": 15,
        "accuracy": 1.0,
        "meta_anns": {
            "Status": {
                "value": "Affirmed",
                "confidence": 0.9999833106994629,
                "name": "Status"
            }
        },
    }]
    model_service.annotate.return_value = annotations
    model_manager = ModelManager(None, None)
    model_manager.model_service = model_service
    cms_globals.model_manager_dep = lambda: model_manager

    response = client.post("/process_jsonl",
                           data="invalid json lines",
                           headers={"Content-Type": "application/x-ndjson"})

    assert response.status_code == 400
    assert response.json() == {"message": "Invalid JSON Lines."}


def test_process_unknown_jsonl_properties(model_service, client):
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 0,
        "end": 15,
        "accuracy": 1.0,
        "meta_anns": {
            "Status": {
                "value": "Affirmed",
                "confidence": 0.9999833106994629,
                "name": "Status"
            }
        },
    }]
    model_service.annotate.return_value = annotations
    model_manager = ModelManager(None, None)
    model_manager.model_service = model_service
    cms_globals.model_manager_dep = lambda: model_manager

    response = client.post("/process_jsonl",
                           data='{"unknown": "doc1", "text": "Spinal stenosis"}\n{"unknown": "doc2", "text": "Spinal stenosis"}',
                           headers={"Content-Type": "application/x-ndjson"})

    assert response.status_code == 400
    assert "Invalid JSON properties found." in response.json()["message"]


def test_warning_on_no_redaction(model_service, client):
    annotations = []
    model_service.annotate.return_value = annotations

    response = client.post("/redact?warn_on_no_redaction=true",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})

    assert response.text == "WARNING: No entities were detected for redaction."


def test_redact_with_encryption(model_service, client):
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 0,
        "end": 15,
        "accuracy": 1.0,
        "meta_anns": {
            "Status": {
                "value": "Affirmed",
                "confidence": 0.9999833106994629,
                "name": "Status"
            }
        },
    }]
    body = {
        "text": "Spinal stenosis",
        "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZSqpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf82E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0TexMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbweQIDAQAB\n-----END PUBLIC KEY-----"
    }
    model_service.annotate.return_value = annotations

    response = client.post("/redact_with_encryption",
                           json=body,
                           headers={"Content-Type": "application/json"})

    assert response.json()["redacted_text"] == "[REDACTED_0]"
    assert len(response.json()["encryptions"]) == 1
    assert response.json()["encryptions"][0]["label"] == "[REDACTED_0]"
    assert isinstance(response.json()["encryptions"][0]["encryption"], str)
    assert len(response.json()["encryptions"][0]["encryption"]) > 0


def test_warning_on_no_encrypted_redaction(model_service, client):
    annotations = []
    body = {
        "text": "Spinal stenosis",
        "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZSqpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf82E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0TexMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbweQIDAQAB\n-----END PUBLIC KEY-----"
    }
    model_service.annotate.return_value = annotations

    response = client.post("/redact_with_encryption?warn_on_no_redaction=true",
                           json=body,
                           headers={"Content-Type": "application/json"})

    assert response.json()["message"] == "WARNING: No entities were detected for redaction."


def test_preview_trainer_export(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/preview_trainer_export", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 4


def test_preview_trainer_export_str(client):
    with open(TRAINER_EXPORT_PATH, "r") as f:
        trainer_export_str = f.read()
        response = client.post("/preview_trainer_export", data={"trainer_export_str": trainer_export_str}, headers={"Content-Type": "application/x-www-form-urlencoded"})

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_project_id(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?project_id=14", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_document_id(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 1


def test_preview_trainer_export_with_project_and_document_ids(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?project_id=14&document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert len(response.text.split("<br/>")) == 1


@pytest.mark.parametrize("pid,did", [(14, 1), (1, 3205)])
def test_preview_trainer_export_on_missing_project_or_document(pid, did, client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(f"/preview_trainer_export?project_id={pid}&document_id={did}", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})

    assert response.status_code == 404
    assert response.json() == {"message": "Cannot find any matching documents to preview"}


def test_train_supervised(model_service, client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/train_supervised", files=[("trainer_export", f)])

    model_service.train_supervised.assert_called()
    assert response.status_code == 202
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_train_unsupervised(model_service, client):
    with tempfile.TemporaryFile("r+b") as f:
        f.write(str.encode("[\"Spinal stenosis\"]"))
        response = client.post("/train_unsupervised", files=[("training_data", f)])

    model_service.train_unsupervised.assert_called()
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_train_metacat(model_service, client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/train_metacat", files=[("trainer_export", f)])

    model_service.train_metacat.assert_called()
    assert response.status_code == 202
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_evaluate_with_trainer_export(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/evaluate", files=[("trainer_export", f)])

    assert response.status_code == 202
    assert response.json()["message"] == "Your evaluation started successfully."
    assert "evaluation_id" in response.json()


def test_sanity_check_with_trainer_export(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/sanity-check", files=[("trainer_export", f)])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,name,precision,recall,f1"


def test_inter_annotator_agreement_scores_per_concept(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_concept", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


@pytest.mark.parametrize("pid_a,pid_b,error_message", [(0, 2, "Cannot find the project with ID: 0"), (1, 3, "Cannot find the project with ID: 3")])
def test_project_not_found_on_getting_iaa_scores(pid_a, pid_b, error_message, client):
    with open(TRAINER_EXPORT_MULTI_PROJS_PATH, "rb") as f:
        response = client.post(f"/iaa-scores?annotator_a_project_id={pid_a}&annotator_b_project_id={pid_b}&scope=per_concept", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"message": error_message}


def test_unknown_scope_on_getting_iaa_scores(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=unknown", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"message": "Unknown scope: \"unknown\""}


def test_inter_annotator_agreement_scores_per_doc(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_document", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "doc_id,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


def test_inter_annotator_agreement_scores_per_span(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_span", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "doc_id,span_start,span_end,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


def test_concat_trainer_exports(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/concat_trainer_exports", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"
    assert len(response.text) == 36918


def test_get_annotation_stats(client):
    with open(TRAINER_EXPORT_PATH, "rb") as f1:
        with open(ANOTHER_TRAINER_EXPORT_PATH, "rb") as f2:
            response = client.post("/annotation-stats", files=[
                ("trainer_export", f1),
                ("trainer_export", f2),
            ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,anno_count,anno_unique_counts,anno_ignorance_counts"


def test_extract_entities_from_text_list_file_as_json_file(model_service, client):
    annotations_list = [
        [{
            "label_name": "Spinal stenosis",
            "label_id": "76107001",
            "start": 0,
            "end": 15,
            "accuracy": 1.0,
            "meta_anns": {
                "Status": {
                    "value": "Affirmed",
                    "confidence": 0.9999833106994629,
                    "name": "Status"
                }
            },
        }]
    ] * 15
    model_service.batch_annotate.return_value = annotations_list

    with open(MULTI_TEXTS_FILE_PATH, "rb") as f:
        response = client.post("/process_bulk_file", files=[("multi_text_file", f)])

    assert isinstance(response, httpx.Response)
    assert json.loads(response.content) == [{
        "text": "Description: Intracerebral hemorrhage (very acute clinical changes occurred immediately).\nCC: Left hand numbness on presentation; then developed lethargy later that day.\nHX: On the day of presentation, this 72 y/o RHM suddenly developed generalized weakness and lightheadedness, and could not rise from a chair. Four hours later he experienced sudden left hand numbness lasting two hours. There were no other associated symptoms except for the generalized weakness and lightheadedness. He denied vertigo.\nHe had been experiencing falling spells without associated LOC up to several times a month for the past year.\nMEDS: procardia SR, Lasix, Ecotrin, KCL, Digoxin, Colace, Coumadin.\nPMH: 1)8/92 evaluation for presyncope (Echocardiogram showed: AV fibrosis/calcification, AV stenosis/insufficiency, MV stenosis with annular calcification and regurgitation, moderate TR, Decreased LV systolic function, severe LAE. MRI brain: focal areas of increased T2 signal in the left cerebellum and in the brainstem probably representing microvascular ischemic disease. IVG (MUGA scan)revealed: global hypokinesis of the LV and biventricular dysfunction, RV ejection Fx 45% and LV ejection Fx 39%. He was subsequently placed on coumadin severe valvular heart disease), 2)HTN, 3)Rheumatic fever and heart disease, 4)COPD, 5)ETOH abuse, 6)colonic polyps, 7)CAD, 8)CHF, 9)Appendectomy, 10)Junctional tachycardia.",
        "annotations": [{
            "label_name": "Spinal stenosis",
            "label_id": "76107001",
            "start": 0,
            "end": 15,
            "accuracy": 1.0,
            "meta_anns": {
                "Status": {
                    "value": "Affirmed",
                    "confidence": 0.9999833106994629,
                    "name": "Status"
                }
            },
        }]
    }] * 15
