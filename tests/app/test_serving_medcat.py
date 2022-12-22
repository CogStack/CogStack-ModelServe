import os
import tempfile
from fastapi.testclient import TestClient
from app.api import get_model_server, get_settings
from app.model_services.medcat_model import MedCATModel
from unittest.mock import create_autospec

model_service = create_autospec(MedCATModel)
get_settings().ENABLE_TRAINING_APIS = "true"
get_settings().DISABLE_UNSUPERVISED_TRAINING = "false"
get_settings().ENABLE_EVALUATION_APIS = "true"
get_settings().ENABLE_PREVIEWS_APIS = "true"
app = get_model_server(lambda: model_service)
client = TestClient(app)


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
        }],
        [{
            "label_name": "Spinal stenosis",
            "label_id": "76107001",
            "start": 1,
            "end": 15,
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
            }]
        },
        {
            "text": "Spinal stenosis",
            "annotations": [{
                "label_name": "Spinal stenosis",
                "label_id": "76107001",
                "start": 1,
                "end": 15,
            }]
        }
    ]


def test_preview():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
        "end": 15,
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/preview",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"


def test_preview_trainer_export():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/preview_trainer_export", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_project_id():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/preview_trainer_export?project_id=14", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 2


def test_preview_trainer_export_with_document_id():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/preview_trainer_export?document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 1


def test_preview_trainer_export_with_project_and_document_ids():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/preview_trainer_export?project_id=14&document_id=3205", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert len(response.text.split("<br/>")) == 1


def test_preview_trainer_export_with_project_id_not_present():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/preview_trainer_export?document_id=1", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 404
    assert response.text == "Cannot find any matching documents to preview"


def test_preview_trainer_export_with_document_id_not_present():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/preview_trainer_export?document_id=1", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 404
    assert response.text == "Cannot find any matching documents to preview"


def test_train_supervised():
    with tempfile.TemporaryFile("r+") as f:
        f.write('{"projects":[{"name":"Project1","id":1,"cuis":"","tuis":"","documents":[{"id":1,"name":"1",' +
                '"text":"Spinal stenosis","last_modified":"","annotations":[{"id":1,"cui":"76107001","start":1,' +
                '"end":15,"validated":true,"correct":true,"deleted":false,"alternative":false,"killed":false,' +
                '"last_modified":"","manually_created":false,"acc":1,"meta_anns":[{"name":"Status","value":"Other",' +
                '"acc":1,"validated":true}]}]}]}]}')
        response = client.post("/train_supervised", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    model_service.train_supervised.assert_called()
    assert response.status_code == 202
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_train_unsupervised():
    with tempfile.TemporaryFile("r+") as f:
        f.write("Spinal stenosis")
        response = client.post("/train_unsupervised", files={"training_data": ("note.txt", f, "multipart/form-data")})
    model_service.train_unsupervised.assert_called()
    assert response.json()["message"] == "Your training started successfully."
    assert "training_id" in response.json()


def test_evaluate_with_trainer_export():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export.json")
    with open(path, "r") as f:
        response = client.post("/evaluate", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "concept,name,precision,recall,f1"


def test_intra_annotator_agreement_scores():
    path = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture", "trainer_export_multi_projs.json")
    with open(path, "r") as f:
        response = client.post("/iaa-scores?annotator_a_project_id=1&annotator_b_project_id=2", files={"trainer_export": ("trainer_export.json", f, "multipart/form-data")})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/csv; charset=utf-8"
    assert response.text.split("\n")[0] == "cui,iaa_percentage,cohens_kappa"

