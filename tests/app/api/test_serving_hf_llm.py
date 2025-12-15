import httpx
import json
import pytest
import app.api.globals as cms_globals

from unittest.mock import create_autospec
from fastapi.testclient import TestClient
from app.api.api import get_generative_server
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel
from app.utils import get_settings

config = get_settings()
config.ENABLE_TRAINING_APIS = "true"
config.DISABLE_UNSUPERVISED_TRAINING = "false"
config.ENABLE_EVALUATION_APIS = "true"
config.ENABLE_PREVIEWS_APIS = "true"
config.AUTH_USER_ENABLED = "false"


@pytest.fixture(scope="function")
def llm_model_service():
    yield create_autospec(HuggingFaceLlmModel)


@pytest.fixture(scope="function")
def llm_app(llm_model_service):
    app = get_generative_server(config, msd_overwritten=lambda: llm_model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    yield app
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def client(llm_model_service):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.generate.return_value = "Yeah."
    llm_model_service.create_embeddings.return_value = [[1.0, 2.0, 3.0]]
    app = get_generative_server(config, msd_overwritten=lambda: llm_model_service)
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    client = TestClient(app)
    yield client
    client.app.dependency_overrides.clear()


def test_generate(client):
    response = client.post(
        "/generate?max_tokens=128&temperature=0.7&top_p=0.9&stop_sequences=end",
        data="Alright?",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 200
    assert response.headers["x-cms-tracking-id"], "x-cms-tracking-id header is missing"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "Yeah."


@pytest.mark.asyncio
async def test_stream_generate(llm_model_service, llm_app):
    llm_model_service.generate_async.return_value = "Fine."
    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/stream/generate?max_tokens=32&temperature=0.7&top_p=0.9&stop_sequences=end",
            data="How are you doing?",
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 200
    assert response.headers["x-cms-tracking-id"], "x-cms-tracking-id header is missing"
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.text == "Fine."


@pytest.mark.asyncio
async def test_generate_chat_completions(llm_model_service, llm_app):
    llm_model_service.generate.return_value = "I'm a chat bot."
    request_data = {
      "messages": [
        {
          "role": "system",
          "content": "You are a chat bot."
        },
        {
          "role": "user",
          "content": "Who are you?"
        }
      ],
      "model": "HuggingFace LLM model",
      "stream": True,
      "max_tokens": 128,
      "temperature": 0.7
    }
    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions?max_tokens=128&temperature=0.7",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.text.startswith("data:")
    assert "id" in response.text
    assert "chat.completion.chunk" in response.text


def test_create_embeddings(client):
    request_data = {
        "input": ["Alright"],
        "model": "HuggingFace LLM model",
    }
    response = client.post(
        "/v1/embeddings",
        data=json.dumps(request_data),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [1.0, 2.0, 3.0], "index": 0}],
        "model": "HuggingFace LLM model"
    }
