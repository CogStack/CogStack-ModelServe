import api.globals as cms_globals
from fastapi.testclient import TestClient
from api.api import get_model_server
from utils import get_settings
from model_services.trf_model_deid import TransformersModelDeIdentification
from unittest.mock import create_autospec

model_service = create_autospec(TransformersModelDeIdentification)
config = get_settings()
config.AUTH_USER_ENABLED = "true"
app = get_model_server(msd_overwritten=lambda: model_service)
client = TestClient(app)
app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None


def test_healthz():
    assert client.get("/healthz").content.decode("utf-8") == "OK"


def test_readyz():
    model_card = {
        "api_version": "0.0.1",
        "model_description": "deid_model_description",
        "model_type": "model_type",
        "model_card": None,
    }
    model_service.info.return_value = model_card
    assert client.get("/readyz").content.decode("utf-8") == "model_type"


def test_info():
    model_card = {
        "api_version": "0.0.1",
        "model_description": "deid_model_description",
        "model_type": "model_type",
        "model_card": None,
    }
    model_service.info.return_value = model_card
    response = client.get("/info")
    assert response.json() == model_card


def test_process():
    annotations = [{
        "label_name": "NW1 2BU",
        "label_id": "C2120",
        "start": 0,
        "end": 6,
    }]
    model_service.annotate.return_value = annotations
    response = client.post("/process",
                           data="NW1 2BU",
                           headers={"Content-Type": "text/plain"})
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
    model_service.batch_annotate.return_value = annotations_list
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


def test_preview():
    annotations = [{
        "label_name": "NW1 2BU",
        "label_id": "C2120",
        "start": 0,
        "end": 6,
    }]
    model_service.annotate.return_value = annotations
    model_service.model_name = "SNOMED Model"
    response = client.post("/preview",
                           data="NW1 2BU",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
