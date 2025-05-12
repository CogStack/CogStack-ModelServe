import pytest
import app.api.globals as cms_globals
from unittest.mock import create_autospec

from fastapi.testclient import TestClient
from app.api.api import get_model_server
from app.utils import get_settings, load_pydantic_object_from_dict
from app.model_services.trf_model_deid import TransformersModelDeIdentification
from app.domain import ModelCard, ModelType, Annotation


config = get_settings()
config.AUTH_USER_ENABLED = "true"


@pytest.fixture(scope="function")
def model_service():
    yield create_autospec(TransformersModelDeIdentification)


@pytest.fixture(scope="function")
def client(model_service):
    app = get_model_server(config, msd_overwritten=lambda: model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    client = TestClient(app)
    yield client
    client.app.dependency_overrides.clear()


def test_healthz(client):
    assert client.get("/healthz").content.decode("utf-8") == "OK"


def test_readyz(model_service, client):
    model_card = load_pydantic_object_from_dict(
        ModelCard,
        {
            "api_version": "0.0.1",
            "model_description": "deid_model_description",
            "model_type": ModelType.TRANSFORMERS_DEID,
            "model_card": None,
            "labels": None,
        },
    )
    model_service.info.return_value = model_card

    assert client.get("/readyz").content.decode("utf-8") == ModelType.TRANSFORMERS_DEID


def test_info(model_service, client):
    raw = {
        "api_version": "0.0.1",
        "model_description": "deid_model_description",
        "model_type": ModelType.TRANSFORMERS_DEID.value,
        "model_card": None,
        "labels": None,
    }
    model_card = load_pydantic_object_from_dict(ModelCard, raw)
    model_service.info.return_value = model_card

    response = client.get("/info")

    assert response.json() == raw


def test_process(model_service, client):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "NW1 2BU",
                "label_id": "C2120",
                "start": 0,
                "end": 6,
            },
        )
    ]
    model_service.annotate.return_value = annotations

    response = client.post(
        "/process",
        data="NW1 2BU",
        headers={"Content-Type": "text/plain"},
    )

    assert response.json() == {
        "text": "NW1 2BU",
        "annotations": [{
            "label_name": "NW1 2BU",
            "label_id": "C2120",
            "start": 0,
            "end": 6,
        }]
    }


def test_process_bulk(model_service, client):
    annotations_list = [
        [
            load_pydantic_object_from_dict(
                Annotation,
                {
                    "label_name": "NW1 2BU",
                    "label_id": "C2120",
                    "start": 0,
                    "end": 6,
                },
            )
        ],
        [
            load_pydantic_object_from_dict(
                Annotation,
                {
                    "label_name": "NW1 2DA",
                    "label_id": "C2120",
                    "start": 0,
                    "end": 6,
                },
            )
        ],
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


def test_preview(model_service, client):
    annotations = [
        load_pydantic_object_from_dict(
            Annotation,
            {
                "label_name": "NW1 2BU",
                "label_id": "C2120",
                "start": 0,
                "end": 6,
            },
        )
    ]
    model_service.annotate.return_value = annotations
    model_service.model_name = "De-Identification Model"

    response = client.post(
        "/preview",
        data="NW1 2BU",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/octet-stream"
