import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.api import get_model_server
from app.utils import get_settings
from app.model_services.medcat_model import MedCATModel
from unittest.mock import create_autospec

model_service = create_autospec(MedCATModel)
get_settings().ENABLE_TRAINING_APIS = "true"
get_settings().DISABLE_UNSUPERVISED_TRAINING = "false"
get_settings().ENABLE_EVALUATION_APIS = "true"
get_settings().ENABLE_PREVIEWS_APIS = "true"
get_settings().AUTH_USER_ENABLED = "false"
app = get_model_server(lambda: model_service)
client = TestClient(app)
TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
ANOTHER_TRAINER_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "another_trainer_export.json")
TRAINER_EXPORT_MULTI_PROJS_PATH = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export_multi_projs.json")


def test_info():
    model_card = {
        "api_version": "0.0.1",
        "model_description": "medcat_model_description",
        "model_type": "model_type",
        "model_card": None,
    }
    model_service.info.return_value = model_card
    response = client.get("/info")
    assert response.json() == model_card


def test_process():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
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
    response = client.post("/process",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.json() == {
        "text": "Spinal stenosis",
        "annotations": annotations
    }


def test_process_bulk():
    annotations_list = [
        [{
            "label_name": "Spinal stenosis",
            "label_id": "76107001",
            "start": 1,
            "end": 15,
            "accuracy": 1.0,
            "meta_anns": {
                "Status": {
                    "value": "Affirmed",
                    "confidence": 0.9999833106994629,
                    "name": "Status"
                }
            },
        }],
        [{
            "label_name": "Spinal stenosis",
            "label_id": "76107001",
            "start": 1,
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
    ]
    model_service.batch_annotate.return_value = annotations_list
    response = client.post("/process_bulk", json=["Spinal stenosis", "Spinal stenosis"])
    assert response.json() == [
        {
            "text": "Spinal stenosis",
            "annotations": [{
                "label_name": "Spinal stenosis",
                "label_id": "76107001",
                "start": 1,
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
        },
        {
            "text": "Spinal stenosis",
            "annotations": [{
                "label_name": "Spinal stenosis",
                "label_id": "76107001",
                "start": 1,
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
        }
    ]


def test_preview():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
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
    response = client.post("/preview",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"


def test_preview_trainer_export():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_project_id():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?project_id=14", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_document_id():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 1


def test_preview_trainer_export_with_project_and_document_ids():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/preview_trainer_export?project_id=14&document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 1


@pytest.mark.parametrize("pid,did", [(14, 1), (1, 3205)])
def test_preview_trainer_export_on_missing_project_or_document(pid, did):
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post(f"/preview_trainer_export?project_id={pid}&document_id={did}", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 404
    assert response.json() == {"detail": "Cannot find any matching documents to preview"}


def test_train_supervised():
    with tempfile.TemporaryFile("r+b") as f:
        f.write(str.encode('{"projects":[{"name":"Project1","id":1,"cuis":"","tuis":"","documents":[{"id":1,"name":"1",' +
                           '"text":"Spinal stenosis","last_modified":"","annotations":[{"id":1,"cui":"76107001","start":1,' +
                           '"end":15,"validated":true,"correct":true,"deleted":false,"alternative":false,"killed":false,' +
                           '"last_modified":"","manually_created":false,"acc":1,"meta_anns":[{"name":"Status","value":"Other",' +
                           '"acc":1,"validated":true}]}]}]}]}'))
        response = client.post("/train_supervised", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    model_service.train_supervised.assert_called()
    assert response.status_code == 202
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_train_unsupervised():
    with tempfile.TemporaryFile("r+b") as f:
        f.write(str.encode("Spinal stenosis"))
        response = client.post("/train_unsupervised", files={"training_data": ("note.txt", f, "multipart/form-data")})
    model_service.train_unsupervised.assert_called()
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_evaluate_with_trainer_export():
    with open(TRAINER_EXPORT_PATH, "rb") as f:
        response = client.post("/evaluate", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,name,precision,recall,f1"


def test_intra_annotator_agreement_scores_per_concept():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_concept", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


@pytest.mark.parametrize("pid_a,pid_b,error_message", [(0, 2, "Cannot find the project with ID: 0"), (1, 3, "Cannot find the project with ID: 3")])
def test_project_not_found_on_getting_iaa_scores(pid_a, pid_b, error_message):
    with open(TRAINER_EXPORT_MULTI_PROJS_PATH, "rb") as f:
        response = client.post(f"/iaa-scores?annotator_a_project_id={pid_a}&annotator_b_project_id={pid_b}&scope=per_concept", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": error_message}


def test_unknown_scope_on_getting_iaa_scores():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=unknown", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "Unknown scope: \"unknown\""}


def test_intra_annotator_agreement_scores_per_doc():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_document", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "doc_id,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


def test_intra_annotator_agreement_scores_per_span():
    response = client.post("/iaa-scores?annotator_a_project_id=14&annotator_b_project_id=15&scope=per_span", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "doc_id,span_start,span_end,iaa_percentage,cohens_kappa,iaa_percentage_meta,cohens_kappa_meta"


def test_concat_trainer_exports():
    response = client.post("/concat_trainer_exports", files=[
        ("trainer_export", open(TRAINER_EXPORT_PATH, "rb")),
        ("trainer_export", open(ANOTHER_TRAINER_EXPORT_PATH, "rb")),
    ])
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"
    assert len(response.text) == 36842
