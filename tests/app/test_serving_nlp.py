from fastapi.testclient import TestClient
from app.serve import get_model_server
from app.model_services.nlp_model import NlpModel
from unittest.mock import create_autospec


model = create_autospec(NlpModel)
client = TestClient(get_model_server(model))


def test_info():
    model.info.return_value = {
        "model_description": "model_description",
        "model_type": "model_type"
    }
    response = client.get("/info")
    assert response.json() == {
        "model_description": "model_description",
        "model_type": "model_type"
    }


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


def test_trainsupervised():
    annotations = {}
    client.post("/trainsupervised", json=annotations)
    model.train_supervised.assert_called_with(annotations)


def test_trainunsupervised():
    texts = ["Spinal stenosis", "Spinal stenosis"]
    client.post("/trainunsupervised", json=texts)
    model.train_unsupervised.assert_called_with(texts)
