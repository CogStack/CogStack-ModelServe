import httpx
import json
import pytest
import app.api.globals as cms_globals

from unittest.mock import create_autospec, Mock
from fastapi.testclient import TestClient
from app.api.api import get_generative_server
from app.model_services.huggingface_llm_model import HuggingFaceLlmModel
from app.domain import GenerationResult
from app.exception import GenerationException
from app.utils import get_settings, dump_pydantic_object_to_dict
from tests.app.helper import disable_rate_limits


config = get_settings()
config.ENABLE_TRAINING_APIS = "true"
config.DISABLE_UNSUPERVISED_TRAINING = "false"
config.ENABLE_EVALUATION_APIS = "true"
config.ENABLE_PREVIEWS_APIS = "true"
config.AUTH_USER_ENABLED = "false"
disable_rate_limits(config)


@pytest.fixture(scope="function")
def llm_model_service():
    model_service = create_autospec(HuggingFaceLlmModel)
    model_service.model = Mock()
    model_service.model.config = Mock()
    model_service.model.config.max_position_embeddings = 2048
    model_service.tokenizer = Mock()
    model_service.tokenizer.chat_template = None
    model_service.tokenizer.default_chat_template = None
    model_service.tokenizer.encode.return_value = []
    yield model_service

@pytest.fixture(scope="function")
def llm_app(llm_model_service):
    app = get_generative_server(
        config, msd_overwritten=lambda: llm_model_service
    )
    app.dependency_overrides[cms_globals.props.current_active_user] = lambda: None
    yield app
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(llm_model_service):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.api_version = "0.0.0"
    llm_model_service.digest = "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    llm_model_service.generate.return_value = GenerationResult(
        text="Yeah.",
        prompt_token_num=1,
        completion_token_num=1,
    )
    llm_model_service.create_embeddings.return_value = [[1.0, 2.0, 3.0]]
    app = get_generative_server(
        config, msd_overwritten=lambda: llm_model_service
    )
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
    async def _gen():
        yield "Fine."
        yield GenerationResult(text="Fine.", prompt_token_num=1, completion_token_num=1)
    llm_model_service.generate_async.return_value = _gen()
    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/stream/generate?max_tokens=32&temperature=0.7&top_p=0.9&stop_sequences=end",
            data="How are you doing?",
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 200
    assert response.headers["x-cms-tracking-id"], "x-cms-tracking-id header is missing"
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert "Fine." in response.text


@pytest.mark.asyncio
async def test_openai_generate_chat_completions(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.generate.return_value = GenerationResult(
        text="I'm a chat bot.",
        prompt_token_num=1,
        completion_token_num=1,
    )
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
      "stream": False,
      "stream_options": {"include_usage": True},
      "max_tokens": 128,
      "temperature": 0.7
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/chat/completions?max_tokens=128&temperature=0.7",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    response_json = response.json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response_json["object"] == "chat.completion"
    assert response_json["model"] == "HuggingFace LLM model"
    assert response_json["choices"][0]["message"]["content"] == "I'm a chat bot."
    assert "usage" in response_json


@pytest.mark.asyncio
async def test_openai_generate_chat_completions_with_tools(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.generate.return_value = GenerationResult(
        text="I'm a chat bot.",
        prompt_token_num=1,
        completion_token_num=1,
    )
    llm_model_service.tokenizer.chat_template = "chat template"
    llm_model_service.tokenizer.apply_chat_template.return_value = "prompt"
    request_data = {
      "messages": [
        {
          "role": "system",
          "content": "You are a chat bot."
        },
        {
          "role": "user",
          "content": "How is the weather in London?"
        }
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
              "type": "object",
              "properties": {
                "city": {"type": "string"}
              },
              "required": ["city"]
            }
          }
        }
      ],
      "model": "HuggingFace LLM model",
      "stream": False,
      "stream_options": {"include_usage": True},
      "max_tokens": 128,
      "temperature": 0.7
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/chat/completions?max_tokens=128&temperature=0.7",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    llm_model_service.generate.assert_called_once()
    assert llm_model_service.tokenizer.apply_chat_template.called
    tools_arg = llm_model_service.tokenizer.apply_chat_template.call_args.kwargs["tools"]
    assert tools_arg == [request_data["tools"][0]]


@pytest.mark.asyncio
async def test_openai_generate_chat_completions_with_response_format(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.generate.return_value = GenerationResult(
        text="{\"age\": 28}",
        prompt_token_num=1,
        completion_token_num=1,
    )
    captured_parser = {}

    def _generate(*_args, **kwargs):
        json_schema_parser = kwargs.get("json_schema_parser")
        if json_schema_parser is not None:
            captured_parser["parser"] = json_schema_parser
        return GenerationResult(
            text="{\"age\": 28}",
            prompt_token_num=1,
            completion_token_num=1,
        )

    llm_model_service.generate.side_effect = _generate
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
        "additionalProperties": False
    }
    request_data = {
      "messages": [
        {
          "role": "user",
          "content": "Extract age from the text: A 28-year-old patient."
        }
      ],
      "response_format": {
        "type": "json_schema",
        "json_schema": {
          "name": "person",
          "schema": schema
        }
      },
      "model": "HuggingFace LLM model",
      "stream": False,
      "stream_options": {"include_usage": True},
      "max_tokens": 128,
      "temperature": 0.7
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/chat/completions?max_tokens=128",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    llm_model_service.generate.assert_called_once()
    assert set(schema.keys()).issubset(
        dump_pydantic_object_to_dict(captured_parser["parser"].context.model_class).keys()
    )


@pytest.mark.asyncio
async def test_openai_generate_chat_completions_stream(llm_model_service, llm_app):
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
      "stream_options": {"include_usage": True},
      "max_tokens": 128,
      "temperature": 0.7
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/chat/completions?max_tokens=128&temperature=0.7",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.text.startswith("data:")
    assert "id" in response.text
    assert "chat.completion.chunk" in response.text


@pytest.mark.asyncio
async def test_openai_generate_chat_completions_stream_without_usage(llm_model_service, llm_app):
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
      "stream_options": {"include_usage": False},
      "max_tokens": 128,
      "temperature": 0.7
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/chat/completions?max_tokens=128&temperature=0.7",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert '"usage"' not in response.text


@pytest.mark.asyncio
async def test_openai_generate_chat_completions_stream_with_generation_exception(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"

    async def _failing_async_gen():
        raise GenerationException("stream failed")
        yield ""

    llm_model_service.generate_async = Mock(return_value=_failing_async_gen())
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
      "stream_options": {"include_usage": True},
      "max_tokens": 128,
      "temperature": 0.7
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/chat/completions?max_tokens=128&temperature=0.7",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert '"type": "generation_error"' in response.text
    assert "[DONE]" in response.text


@pytest.mark.asyncio
async def test_openai_generate_completions(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.generate.return_value = GenerationResult(
        text="I'm a chat bot.",
        prompt_token_num=1,
        completion_token_num=1,
    )
    request_data = {
        "model": "HuggingFace LLM model",
        "prompt": "Who are you?",
        "stream_options": {"include_usage": True},
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False,
    }

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/completions",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    response_json = response.json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response_json["object"] == "text_completion"
    assert response_json["model"] == "HuggingFace LLM model"
    assert response_json["choices"][0]["text"] == "I'm a chat bot."


@pytest.mark.asyncio
async def test_openai_generate_completions_stream(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"

    async def async_gen():
        yield "I'm a chat bot."

    llm_model_service.generate_async.return_value = async_gen()
    request_data = {
        "model": "HuggingFace LLM model",
        "prompt": "Who are you?",
        "stream_options": {"include_usage": True},
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": True,
    }
    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/completions",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.text.startswith("data:")
    assert "id" in response.text
    assert "text_completion" in response.text
    assert "[DONE]" in response.text


@pytest.mark.asyncio
async def test_openai_generate_completions_stream_with_no_usage(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"

    async def async_gen():
        yield "I'm a chat bot."

    llm_model_service.generate_async.return_value = async_gen()
    request_data = {
        "model": "HuggingFace LLM model",
        "prompt": "Who are you?",
        "stream_options": {"include_usage": False},
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": True,
    }
    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/openai/v1/completions",
            data=json.dumps(request_data),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert '"usage"' not in response.text


def test_openai_create_embeddings(client):
    request_data = {
        "input": ["Alright"],
        "model": "HuggingFace LLM model",
    }
    response = client.post(
        "/openai/v1/embeddings",
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


def test_openai_list_models(client):
    response = client.get("/openai/v1/models")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    response_json = response.json()
    assert response_json["object"] == "list"
    assert len(response_json["data"]) == 1
    assert response_json["data"][0]["id"] == "HuggingFace_LLM_model"
    assert response_json["data"][0]["object"] == "model"
    assert response_json["data"][0]["created"] == 0
    assert response_json["data"][0]["owned_by"] == "cms"


def test_openai_get_model(client):
    response = client.get("/openai/v1/models/HuggingFace_LLM_model")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    response_json = response.json()
    assert response_json["id"] == "HuggingFace_LLM_model"
    assert response_json["object"] == "model"
    assert response_json["created"] == 0
    assert response_json["owned_by"] == "cms"
    assert response_json["permission"] == []
    assert response_json["root"] == "HuggingFace_LLM_model"
    assert response_json["parent"] is None


def test_ollama_health(client):
    get_response = client.get("/ollama")
    head_response = client.head("/ollama")

    assert get_response.status_code == 200
    assert get_response.json() == {"status": "ok"}
    assert head_response.status_code == 200


def test_ollama_version(client):
    response = client.get("/ollama/api/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_ollama_tags(client, llm_model_service):
    llm_model_service.model = Mock()
    llm_model_service.model.config = Mock()
    llm_model_service.model.config.model_type = "model_type"
    llm_model_service.tokenizer = Mock(chat_template="chat template")
    llm_model_service.info.return_value = Mock(model_card={"model_type": "model_type"})

    tags_response = client.get("/ollama/api/tags")

    assert tags_response.status_code == 200
    assert tags_response.json()["models"][0]["name"] == "HuggingFace_LLM_model"


def test_ollama_show(client, llm_model_service):
    llm_model_service.model_name = "HuggingFace LLM model"
    llm_model_service.model = Mock()
    llm_model_service.model.config = Mock()
    llm_model_service.model.config.model_type = "model_type"
    llm_model_service.tokenizer = Mock(chat_template="chat template")
    llm_model_service.info.return_value = Mock(model_card={"model_type": "model_type"})

    response = client.post(
        "/ollama/api/show",
        data=json.dumps({"model": "HuggingFace_LLM_model"}),
        headers={"Content-Type": "application/json"},
    )

    response_json = response.json()
    assert response.status_code == 200
    assert response_json["model_info"]["model_type"] == "model_type"
    assert response_json["modelfile"] == "HuggingFace LLM model"
    assert response_json["template"] == "chat template"
    assert response_json["details"]["family"] == "model_type"
    assert response_json["capabilities"] == ["completion", "chat", "create_embeddings"]


def test_ollama_generate(client):
    response = client.post(
        "/ollama/api/generate",
        data=json.dumps({
            "model": "HuggingFace_LLM_model",
            "prompt": "Hello",
            "stream": False,
            "options": {"num_predict": 16, "temperature": 0.2, "top_p": 0.8},
        }),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["model"] == "HuggingFace_LLM_model"
    assert response_json["response"] == "Yeah."
    assert response_json["done"] is True


def test_ollama_generate_with_format(client, llm_model_service):
    captured_parser = {}

    def _generate(*_args, **kwargs):
        json_schema_parser = kwargs.get("json_schema_parser")
        if json_schema_parser is not None:
            captured_parser["parser"] = json_schema_parser
        return GenerationResult(
            text="{\"age\": 28}",
            prompt_token_num=1,
            completion_token_num=1,
        )

    llm_model_service.generate.side_effect = _generate
    format = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
        "additionalProperties": False,
    }

    response = client.post(
        "/ollama/api/generate",
        data=json.dumps({
            "model": "HuggingFace_LLM_model",
            "prompt": "Extract age",
            "stream": False,
            "format": format,
        }),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert set(format.keys()).issubset(
        dump_pydantic_object_to_dict(captured_parser["parser"].context.model_class).keys()
    )


@pytest.mark.asyncio
async def test_ollama_generate_stream(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"

    async def async_gen():
        yield "hi "
        yield "there"
    llm_model_service.generate_async = Mock(return_value=async_gen())

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/ollama/api/generate",
            data=json.dumps({
                "model": "HuggingFace_LLM_model",
                "prompt": "Hello",
                "stream": True,
            }),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert '"done": false' in response.text.lower()
    assert '"done": true' in response.text.lower()


@pytest.mark.asyncio
async def test_ollama_generate_stream_with_generation_exception(llm_model_service, llm_app):
    llm_model_service.model_name = "HuggingFace LLM model"

    async def _failing_async_gen():
        raise GenerationException("stream failed")
        yield ""

    llm_model_service.generate_async = Mock(return_value=_failing_async_gen())

    async with httpx.AsyncClient(app=llm_app, base_url="http://test") as ac:
        response = await ac.post(
            "/ollama/api/generate",
            data=json.dumps({
                "model": "HuggingFace_LLM_model",
                "prompt": "Hello",
                "stream": True,
            }),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert '"done_reason": "error"' in response.text
    assert '"error": "stream failed"' in response.text


def test_ollama_chat(client):
    chat_response = client.post(
        "/ollama/api/chat",
        data=json.dumps({
            "model": "HuggingFace_LLM_model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }),
        headers={"Content-Type": "application/json"},
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["message"]["role"] == "assistant"


def test_ollama_chat_with_format(client, llm_model_service):
    captured_parser = {}

    def _generate(*_args, **kwargs):
        json_schema_parser = kwargs.get("json_schema_parser")
        if json_schema_parser is not None:
            captured_parser["parser"] = json_schema_parser
        return GenerationResult(
            text="{\"age\": 28}",
            prompt_token_num=1,
            completion_token_num=1,
        )

    llm_model_service.generate.side_effect = _generate
    format = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
        "additionalProperties": False,
    }

    chat_response = client.post(
        "/ollama/api/chat",
        data=json.dumps({
            "model": "HuggingFace_LLM_model",
            "messages": [{"role": "user", "content": "Extract age"}],
            "stream": False,
            "format": format,
        }),
        headers={"Content-Type": "application/json"},
    )

    assert chat_response.status_code == 200
    assert set(format.keys()).issubset(
        dump_pydantic_object_to_dict(captured_parser["parser"].context.model_class).keys()
    )


def test_ollama_embed(client):
    response = client.post(
        "/ollama/api/embed",
        data=json.dumps({
            "model": "HuggingFace_LLM_model",
            "input": ["test", "blah"],
        }),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert "embeddings" in response.json()
