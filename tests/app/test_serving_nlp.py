import pytest
from fastapi.testclient import TestClient
from app.serve import get_model_server
from app.model_services.medcat_model import MedCATModel
from unittest.mock import create_autospec


model = create_autospec(MedCATModel)
client = TestClient(get_model_server(model))


def test_info():
    model_card = {
        "api_version": "0.0.1",
        "model_description": "model_description",
        "model_type": "model_type",
        "model_card": None,
    }
    model.info.return_value = model_card
    response = client.get("/info")
    assert response.json() == model_card


def test_process():
    annotations = [{
        "label_name": "Spinal stenosis",
        "label_id": "76107001",
        "start": 1,
        "end": 15,
    }]
    model.annotate.return_value = annotations
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
    model.batch_annotate.return_value = annotations_list
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
    model.annotate.return_value = annotations
    response = client.post("/preview",
                           data="Spinal stenosis",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 201
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"


@pytest.mark.skip()
def test_train_supervised():
    annotations = {}
    client.post("/train_supervised", json=annotations)
    model.train_supervised.assert_called_with(annotations)


@pytest.mark.skip()
def test_train_unsupervised():
    texts = ["Spinal stenosis", "Spinal stenosis"]
    client.post("/train_unsupervised", json=texts)
    model.train_unsupervised.assert_called_with(texts)
