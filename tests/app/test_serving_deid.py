from fastapi.testclient import TestClient
from app.serve import get_model_server
from app.model_services.deid_model import DeIdModel
from unittest.mock import create_autospec


model = create_autospec(DeIdModel)
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
        "label_name": "NW1 2BU",
        "label_id": "C2120",
        "start": 0,
        "end": 6,
    }]
    model.annotate.return_value = annotations
    response = client.post("/process?text=NW1%202BU")
    assert response.json() == {
        "text": "NW1 2BU",
        "annotations": annotations
    }


def test_process_bulk():
    annotations_list = [
        [{
            "label_name": "NW1 2BU",
            "label_id": "C2120",
            "start": 0,
            "end": 6,
        }],
        [{
            "label_name": "NW1 2DA",
            "label_id": "C2120",
            "start": 0,
            "end": 6,
        }]
    ]
    model.batch_annotate.return_value = annotations_list
    response = client.post("/process_bulk", json=["NW1 2BU", "NW1 2DA"])
    assert response.json() == [
        {
            "text": "NW1 2BU",
            "annotations": [{
                "label_name": "NW1 2BU",
                "label_id": "C2120",
                "start": 0,
                "end": 6,
            }]
        },
        {
            "text": "NW1 2DA",
            "annotations": [{
                "label_name": "NW1 2DA",
                "label_id": "C2120",
                "start": 0,
                "end": 6,
            }]
        }
    ]
