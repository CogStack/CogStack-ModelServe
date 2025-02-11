import httpx
import json
import pytest

import app.api.globals as cms_globals

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from unittest.mock import create_autospec
from app.api.api import get_stream_server
from app.utils import get_settings
from app.model_services.medcat_model import MedCATModel
from app.management.model_manager import ModelManager


config = get_settings()
config.ENABLE_TRAINING_APIS = "true"
config.DISABLE_UNSUPERVISED_TRAINING = "false"
config.ENABLE_EVALUATION_APIS = "true"
config.ENABLE_PREVIEWS_APIS = "true"
config.AUTH_USER_ENABLED = "false"


@pytest.fixture(scope="function")
def model_service():
    yield create_autospec(MedCATModel)


@pytest.fixture(scope="function")
def app(model_service):
    app = get_stream_server(msd_overwritten=lambda: model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    yield app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_stream_process_empty_stream(model_service, app):
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/stream/process", data="", headers={"Content-Type": "application/x-ndjson"})

    assert response.status_code == 200
    jsonlines = b""
    async for chunk in response.aiter_bytes():
        jsonlines += chunk
    assert json.loads(jsonlines.decode("utf-8").splitlines()[-1])["error"] == "Empty stream"


@pytest.mark.asyncio
async def test_stream_process_invalidate_jsonl(model_service, app):
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/stream/process",
                                 data='{"name": "doc1", "text": Spinal stenosis}\n'.encode("utf-8"),
                                 headers={"Content-Type": "application/x-ndjson"})

    assert response.status_code == 200
    jsonlines = b""
    async for chunk in response.aiter_bytes():
        jsonlines += chunk
    assert json.loads(jsonlines.decode("utf-8").splitlines()[-1])["error"] == "Invalid JSON Line"


@pytest.mark.asyncio
async def test_stream_process_unknown_jsonl_property(model_service, app):
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/stream/process",
                                 data='{"unknown": "doc1", "text": "Spinal stenosis"}\n{"unknown": "doc2", "text": "Spinal stenosis"}',
                                 headers={"Content-Type": "application/x-ndjson"})

    assert response.status_code == 200
    jsonlines = b""
    async for chunk in response.aiter_bytes():
        jsonlines += chunk
    assert "Invalid JSON properties found" in json.loads(jsonlines.decode("utf-8").splitlines()[-1])["error"]


def test_websocket_process_on_annotation_error(model_service, app):
    model_service.async_annotate.side_effect = Exception("something went wrong")
    model_manager = ModelManager(None, None)
    model_manager.model_service = model_service
    cms_globals.model_manager_dep = lambda: model_manager

    with pytest.raises(WebSocketDisconnect):
        with TestClient(app) as client:
            with client.websocket_connect("/stream/ws") as websocket:
                websocket.send_text("Spinal stenosis")
                response = websocket.receive_text()
                assert response == "ERROR: something went wrong"
