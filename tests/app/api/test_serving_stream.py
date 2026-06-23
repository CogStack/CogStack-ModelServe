import asyncio
import httpx
import json
import pytest

import app.api.globals as cms_globals

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from unittest.mock import create_autospec, AsyncMock
from app.api.api import get_stream_server
from app.utils import get_settings
from app.model_services.medcat_model import MedCATModel
from app.management.model_manager import ModelManager
from app.domain import Annotation
from tests.app.helper import disable_rate_limits


@pytest.fixture(scope="function")
def ner_model_service():
    return create_autospec(MedCATModel)


@pytest.fixture(scope="function")
def ner_app(ner_model_service):
    config = get_settings()
    config.ENABLE_TRAINING_APIS = "true"
    config.DISABLE_UNSUPERVISED_TRAINING = "false"
    config.ENABLE_EVALUATION_APIS = "true"
    config.ENABLE_PREVIEWS_APIS = "true"
    config.AUTH_USER_ENABLED = "false"
    config.AUTH_USER_ENABLED = "false"
    disable_rate_limits(config)
    app = get_stream_server(config, msd_overwritten=lambda: ner_model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    yield app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_stream_process(ner_app):
    async with httpx.AsyncClient(app=ner_app, base_url="http://test") as ac:
        response = await ac.post(
            "/stream/process",
            data='{ "text": "This is a test"}',
            headers={"Content-Type": "application/x-ndjson"},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_stream_process_empty_stream(ner_app):
    async with httpx.AsyncClient(app=ner_app, base_url="http://test") as ac:
        response = await ac.post("/stream/process", data="", headers={"Content-Type": "application/x-ndjson"})

    assert response.status_code == 200
    jsonlines = b""
    async for chunk in response.aiter_bytes():
        jsonlines += chunk
    assert json.loads(jsonlines.decode("utf-8").splitlines()[-1])["error"] == "Empty stream"


@pytest.mark.asyncio
async def test_stream_process_invalidate_jsonl(ner_app):
    async with httpx.AsyncClient(app=ner_app, base_url="http://test") as ac:
        response = await ac.post(
            "/stream/process",
            data='{"name": "doc1", "text": Spinal stenosis}\n'.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )

    assert response.status_code == 200
    jsonlines = b""
    async for chunk in response.aiter_bytes():
        jsonlines += chunk
    assert json.loads(jsonlines.decode("utf-8").splitlines()[-1])["error"] == "Invalid JSON Line"


@pytest.mark.asyncio
async def test_stream_process_unknown_jsonl_property(ner_app):
    async with httpx.AsyncClient(app=ner_app, base_url="http://test") as ac:
        response = await ac.post(
            "/stream/process",
            data='{"unknown": "doc1", "text": "Spinal stenosis"}\n{"unknown": "doc2", "text": "Spinal stenosis"}',
            headers={"Content-Type": "application/x-ndjson"},
        )

    assert response.status_code == 200
    jsonlines = b""
    async for chunk in response.aiter_bytes():
        jsonlines += chunk
    assert "Invalid JSON properties found" in json.loads(jsonlines.decode("utf-8").splitlines()[-1])["error"]


def test_websocket_info(ner_app):
    with TestClient(ner_app) as client:
        response = client.get("/stream/ws", headers={"x-forwarded-prefix": "/cms/medcat"})

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["protocol"] == "WebSocket"
    assert "message" in response_json
    assert "example" in response_json


def test_websocket_process_on_annotation_error(ner_model_service, ner_app):
    ner_model_service.annotate_async.side_effect = Exception("something went wrong")
    model_manager = ModelManager(None, None)
    model_manager.model_service = ner_model_service
    cms_globals.model_manager_dep = lambda: model_manager

    with TestClient(ner_app) as client:
        with client.websocket_connect("/stream/ws") as websocket:
            websocket.send_text("Spinal stenosis")
            response = websocket.receive_text()
            assert response == "ERROR: something went wrong"


@pytest.mark.skip()
@pytest.mark.asyncio
async def test_sse_process(ner_app, ner_model_service):
    fake_annotation = Annotation(
        label_id="C123456",
        label_name="test",
        start=0,
        end=4,
        confidence=0.9,
        doc_name="0",
    )
    ner_model_service.annotate_async = AsyncMock(return_value=[fake_annotation])

    async with (
        httpx.AsyncClient(app=ner_app, base_url="http://test", timeout=30.0) as sse_client,
        httpx.AsyncClient(app=ner_app, base_url="http://test", timeout=30.0) as post_client,
    ):
        sse_data = []
        connection_established = asyncio.Event()

        async def consume_sse():
            async with sse_client.stream(
                "GET", "/stream/sse/events?client_id=abc"
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith(": connected"):
                        connection_established.set()
                        continue

                    if line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        sse_data.append(payload)

                        if payload.get("status") == "all_completed":
                            break

        async def send_data():
            await asyncio.wait_for(connection_established.wait(), timeout=5)

            response = await post_client.post(
                "/stream/sse/process?client_id=abc",
                content='{"text": "This is a test"}\n',
                headers={"Content-Type": "application/x-ndjson"},
            )

            assert response.status_code == 202

        await asyncio.wait_for(
            asyncio.gather(consume_sse(), send_data()),
            timeout=5.0,
        )

        assert len(sse_data) > 0, "No SSE events received"

        statuses = [event.get("status") for event in sse_data if "status" in event]
        assert "started" in statuses
        assert "completed" in statuses
        assert "all_completed" in statuses

        annotations = [
            event["data"] for event in sse_data if event.get("type") == "annotation"
        ]

        assert len(annotations) == 1
        assert annotations[0]["label_id"] == "C123456"
        assert annotations[0]["label_name"] == "test"
