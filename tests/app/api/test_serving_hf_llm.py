import httpx
import pytest
import app.api.globals as cms_globals

from unittest.mock import create_autospec
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


@pytest.mark.asyncio
async def test_stream_generate(llm_model_service, llm_app):
    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/stream/generate?max_tokens=32",
            data="How are you doing?",
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 200