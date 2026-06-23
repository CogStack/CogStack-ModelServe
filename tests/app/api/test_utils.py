import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from app import __version__ as app_version
from app.domain import ModelType
from app.utils import get_settings
from app.api.utils import (
    add_exception_handlers,
    add_rate_limiter,
    get_rate_limiter,
    encrypt,
    decrypt,
    ForwardedPrefixMiddleware,
    init_vllm_engine,
    init_sglang_engine,
)
import sys
import types
import contextlib


def test_add_exception_handlers():
    app = FastAPI()
    add_exception_handlers(app)
    handlers = [handler.__name__ for handler in app.exception_handlers.values()]
    assert "json_decoding_exception_handler" in handlers
    assert "rate_limit_exceeded_handler" in handlers
    assert "start_training_exception_handler" in handlers
    assert "annotation_exception_handler" in handlers
    assert "configuration_exception_handler" in handlers
    assert "unhandled_exception_handler" in handlers


def test_add_middlewares():
    app = FastAPI()
    add_rate_limiter(app, get_settings())
    middlewares = [str(middleware) for middleware in app.user_middleware]
    assert "Middleware(SlowAPIMiddleware)" in middlewares

    streamable_app = FastAPI()
    add_rate_limiter(streamable_app, get_settings(), True)
    middlewares = [str(middleware) for middleware in streamable_app.user_middleware]
    assert "Middleware(SlowAPIASGIMiddleware)" in middlewares


def test_get_per_address_rate_limiter():
    limiter = get_rate_limiter(get_settings(), auth_user_enabled=False)
    assert limiter._key_func.__name__ == "get_remote_address"


def test_get_per_user_rate_limiter():
    limiter = get_rate_limiter(get_settings(), auth_user_enabled=True)
    key_func = limiter._key_func
    assert key_func.__name__ == "_get_user_auth"


def test_encrypt():
    fake_public_key_pem = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3ITkTP8Tm/5FygcwY2EQ
7LgVsuCF0OH7psUqvlXnOPNCfX86CobHBiSFjG9o5ZeajPtTXaf1thUodgpJZVZS
qpVTXwGKo8r0COMO87IcwYigkZZgG/WmZgoZART+AA0+JvjFGxflJAxSv7puGlf8
2E+u5Wz2psLBSDO5qrnmaDZTvPh5eX84cocahVVI7X09/kI+sZiKauM69yoy1bdx
16YIIeNm0M9qqS3tTrjouQiJfZ8jUKSZ44Na/81LMVw5O46+5GvwD+OsR43kQ0Te
xMwgtHxQQsiXLWHCDNy2ZzkzukDYRwA3V2lwVjtQN0WjxHg24BTBDBM+v7iQ7cbw
eQIDAQAB
-----END PUBLIC KEY-----"""
    encrypted = encrypt("test", fake_public_key_pem)
    assert isinstance(encrypted, str)
    assert len(encrypted) > 0


def test_decrypt():
    fake_private_key_pem = """-----BEGIN PRIVATE KEY-----
MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDchORM/xOb/kXK
BzBjYRDsuBWy4IXQ4fumxSq+Vec480J9fzoKhscGJIWMb2jll5qM+1Ndp/W2FSh2
CkllVlKqlVNfAYqjyvQI4w7zshzBiKCRlmAb9aZmChkBFP4ADT4m+MUbF+UkDFK/
um4aV/zYT67lbPamwsFIM7mqueZoNlO8+Hl5fzhyhxqFVUjtfT3+Qj6xmIpq4zr3
KjLVt3HXpggh42bQz2qpLe1OuOi5CIl9nyNQpJnjg1r/zUsxXDk7jr7ka/AP46xH
jeRDRN7EzCC0fFBCyJctYcIM3LZnOTO6QNhHADdXaXBWO1A3RaPEeDbgFMEMEz6/
uJDtxvB5AgMBAAECggEABLc80J610yStZmQf90gYng9Tu3cMtYpXoNnfj6Fzp+af
2eIyIg5+zBVU28t4IUzMK86mGj8gxIuQSXHv3uBpNSerWEFGrzkEXfpJFBIPhl3/
HQ3rsT1gGReHMFw8EFE4LoosYOdyaYJv9JSujRarnA6cLWDWp3tLudkNU+bU1A6n
MyXwM1jyM5RkLKSY5tTuzNZ3fL/Yz+Spuxw9yKFE6l6Rcb0weLYMNVrPlSr4SfJ3
R9WyfRKqO2WXZCJ5sGEOx30Zas6ivsorVZ+b9VWkAaDvCpcbg4ahyfGjhWFWFpCo
+zxFlmfGyouY8OtL7Tq7QSnHxoFvMBv7p/CpTuezDwKBgQDrWGjGsAZrD9sIKuYC
yAo7SkaN8s1tm224MYlFd26vzvVxMUv2ZYgRGDPD3L8KDgzIPpU9ltnyPnKmso6c
92+Uit3p1lCLvrRZI+ArYaXkk7pl/XjAd9FNzIWp5mBCOIeEdpeOpBscaOe1yxDG
VvK1RKBqZNX1vkmcjSSRA6So1wKBgQDv32A76d4UQNzjIQeDn/Q4LGZOKPMyC+ys
u/Pf91hGnu6LvcmKjs2HhgOUlH1Nd5voR+bb0AxbdrOV8EtoYoWAg8c5t/jzWspK
UXIRe37EQeKSV6MwU+93Tcjr2fohdGznc6etECa8b9n05qLZa6pt7MtMM2vI69mR
aCGbtnB3LwKBgQDWUeLI3dBae0v6OibQ7Z7zs4ZhCnYtlNfsX6Ak1MjF7fDyrfQB
ZSDugF3TxhlrbLQTP3rlZZUA2AHM8NqS83p3iabhpjwfpwHSE6u3letfJ3EeJCBt
FjBTaydmO9f5NkWjSeRnD+dojdhFY7HZDaFlliOIAGAgtLOQj7B3JxwybQKBgQDc
bwh+xqJhNmJHD5laKmpCHPs/JH6pJTAwZODult02uOM65AQMIsNZoZw0tGiaAiry
QPE0W3KfsuvCBHsnyDIrMe6pahmLeYmg1kvfKQAL1wghuAutY9USbBcSNtSYXeee
ozgZ4FfYn2lKl5BcAYczUYJZ2n9YuvTLnUgVUojz3QKBgQDmewPhaqYJOKDHeY6D
QySZIZwb2mZd3nozPMzBJuTh5QK+KPkzSeJTihuIZh8ZImD0LX3TX8KSdz9oZQQR
cExDsxcGU7ZcTO9WVwDhqF/9ofkXfLOFKxugLNEA5RA3gRcpCxMRLS4k6dfN9N9o
3RQZkF/usTTvyvFQR96frZb2FQ==
-----END PRIVATE KEY-----"""
    encrypted = "TLlMBh4GDf3BSsO/RKlqG5H7Sxv7OXGbl8qE/6YLQPm3coBbnrRRReX7pLamnjLPUU0PtIRIg2H/hWBWE/3cRtXDPT7jMtmGHMIPO/95A0DkrndIkOeQ29J6TBPBBG6YqBNRb2dyhDBwDIEDjPTiRe68sYz4KkxzSOkcz31314kSkZvdIDtQOgeRDa0/7U0VrJePL2N7SJvEiHf4Xa3vW3/20S3O8s/Yp0Azb/kS9dFa54VO1fNNhJ46OtPpdekiFDR5yvQfHwFVeSDdY+eAuYLTWa6bz/LrQkRAdRi9EW5Iz/q8WgKhZXQJfcXtiKfVuFar2N2KodY7C/45vMOfvw=="
    decrypted = decrypt(encrypted, fake_private_key_pem)
    assert decrypted == "test"


def test_forwarded_prefix_middleware():
    app = FastAPI()
    app.add_middleware(ForwardedPrefixMiddleware)

    @app.get("/ping")
    async def ping(request: Request) -> dict:
        return {"root_path": request.scope.get("root_path")}

    with TestClient(app) as client:
        response = client.get("/ping", headers={"x-forwarded-prefix": "/cms/huggingface-llm"})

    assert response.status_code == 200
    assert response.json() == {"root_path": "/cms/huggingface-llm"}


@pytest.mark.asyncio
async def test_init_vllm_engine(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_utils = types.ModuleType("vllm.utils")
    fake_argparse_utils = types.ModuleType("vllm.utils.argparse_utils")
    fake_engine = types.ModuleType("vllm.engine")
    fake_engine_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    fake_entrypoints = types.ModuleType("vllm.entrypoints")
    fake_openai = types.ModuleType("vllm.entrypoints.openai")
    fake_cli_args = types.ModuleType("vllm.entrypoints.openai.cli_args")
    fake_chat_utils = types.ModuleType("vllm.entrypoints.chat_utils")
    fake_api_server = types.ModuleType("vllm.entrypoints.openai.api_server")

    class FlexibleArgumentParser:
        def parse_args(self, _):
            return types.SimpleNamespace()

    def make_arg_parser(parser):
        return parser

    def validate_parsed_serve_args(args):
        return args

    class AsyncEngineArgs:
        @classmethod
        def from_cli_args(cls, _args):
            return cls()

    async def parse_chat_messages(messages, model_config, content_format="string"):
        return messages, None, None

    def apply_hf_chat_template(*_args, **_kwargs):
        return "prompt"

    async def show_available_models(_request):
        return JSONResponse(content={"data": []})

    async def show_version():
        return JSONResponse(content={"version": "test"})

    async def create_chat_completion(_request, _raw_request):
        return JSONResponse(content={"choices": []})

    async def create_completion(_request, _raw_request):
        return JSONResponse(content={"choices": []})

    async def create_responses(_request, _raw_request):
        return JSONResponse(content={"output": []})

    async def retrieve_responses(_response_id, _raw_request):
        return JSONResponse(content={"output": []})

    async def create_messages(_request, _raw_request):
        return JSONResponse(content={"messages": []})

    async def create_transcriptions(_request, _raw_request):
        return JSONResponse(content={"text": ""})

    async def create_translations(_request, _raw_request):
        return JSONResponse(content={"text": ""})

    @contextlib.asynccontextmanager
    async def build_async_engine_client_from_engine_args(*_args, **_kwargs):
        class _Engine:
            vllm_config = object()
            model_config = types.SimpleNamespace(to_dict=lambda: {"name_or_path": "model_path"})

            async def get_tokenizer(self):
                class _Tokenizer:
                    chat_template = None
                    default_chat_template = None

                    def __call__(self, text, add_special_tokens=False):
                        return types.SimpleNamespace(input_ids=[1, 2, 3])

                return _Tokenizer()

            async def get_model_config(self):
                return self.model_config

        yield _Engine()

    async def init_app_state(_engine, _state, _args):
        return None

    class SamplingParams:
        def __init__(self, max_tokens=0):
            self.max_tokens = max_tokens

    class TokensPrompt(dict):
        def __init__(self, prompt_token_ids=None):
            super().__init__(prompt_token_ids=prompt_token_ids)

    fake_argparse_utils.FlexibleArgumentParser = FlexibleArgumentParser
    fake_cli_args.make_arg_parser = make_arg_parser
    fake_cli_args.validate_parsed_serve_args = validate_parsed_serve_args
    fake_engine_arg_utils.AsyncEngineArgs = AsyncEngineArgs
    fake_chat_utils.parse_chat_messages = parse_chat_messages
    fake_chat_utils.apply_hf_chat_template = apply_hf_chat_template
    fake_api_server.create_chat_completion = create_chat_completion
    fake_api_server.create_completion = create_completion
    fake_api_server.create_responses = create_responses
    fake_api_server.retrieve_responses = retrieve_responses
    fake_api_server.create_messages = create_messages
    fake_api_server.create_transcriptions = create_transcriptions
    fake_api_server.create_translations = create_translations
    fake_api_server.show_available_models = show_available_models
    fake_api_server.show_version = show_version
    fake_api_server.build_async_engine_client_from_engine_args = build_async_engine_client_from_engine_args
    fake_api_server.init_app_state = init_app_state
    fake_vllm.SamplingParams = SamplingParams
    fake_vllm.TokensPrompt = TokensPrompt

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.argparse_utils", fake_argparse_utils)
    monkeypatch.setitem(sys.modules, "vllm.engine", fake_engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", fake_engine_arg_utils)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", fake_entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", fake_openai)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.cli_args", fake_cli_args)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", fake_chat_utils)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.api_server", fake_api_server)

    app = FastAPI()
    app = await init_vllm_engine(app, get_settings(), "model_path", "model_name", "info")
    paths = {route.path for route in app.router.routes}
    assert "/info" in paths
    assert "/generate" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
    assert "/v1/models" in paths
    assert "/v1/version" in paths
    with TestClient(app) as client:
        response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == app_version
    assert data["model_description"] == "model_name"
    assert data["model_type"] == ModelType.HUGGINGFACE_LLM.value
    assert data["model_card"] == {"name_or_path": "model_path"}


@pytest.mark.asyncio
async def test_init_sglang_engine(monkeypatch):
    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_entrypoints = types.ModuleType("sglang.srt.entrypoints")
    fake_engine_mod = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
    fake_server_args_mod = types.ModuleType("sglang.srt.server_args")
    fake_openai_mod = types.ModuleType("sglang.srt.entrypoints.openai")
    fake_serving_chat = types.ModuleType("sglang.srt.entrypoints.openai.serving_chat")
    fake_serving_completions = types.ModuleType("sglang.srt.entrypoints.openai.serving_completions")
    fake_serving_embedding = types.ModuleType("sglang.srt.entrypoints.openai.serving_embedding")
    fake_serving_score = types.ModuleType("sglang.srt.entrypoints.openai.serving_score")
    fake_serving_rerank = types.ModuleType("sglang.srt.entrypoints.openai.serving_rerank")
    fake_metrics = types.ModuleType("sglang.srt.metrics")
    fake_func_timer = types.ModuleType("sglang.srt.metrics.func_timer")
    fake_version = types.ModuleType("sglang.version")
    fake_utils_mod = types.ModuleType("sglang.srt.utils")

    class _ServerArgs:
        model_path = "model_path"
        served_model_name = "model_name"
        log_level = "info"
        log_level_http = "info"
        tokenizer_worker_num = 1
        skip_server_warmup = False
        quantization = None
        model_impl = "transformers"
        mem_fraction_static = 0.9
        api_key = None
        enable_metrics = False

    def prepare_server_args(_argv):
        return _ServerArgs()

    class _TokenizerManager:
        _subprocess_watchdog = None
        model_config = types.SimpleNamespace(to_dict=lambda: {"name_or_path": "model_path"})

    class _TemplateManager:
        pass

    def _launch_subprocesses(server_args, init_tokenizer_manager_func, run_scheduler_process_func, run_detokenizer_process_func):
        return _TokenizerManager(), _TemplateManager(), [object()], object()

    def init_tokenizer_manager(*_a, **_kw): pass
    def run_detokenizer_process(*_a, **_kw): pass
    def run_scheduler_process(*_a, **_kw): pass

    class _GlobalState:
        def __init__(self, tokenizer_manager, template_manager, scheduler_info): pass

    def set_global_state(_gs): pass

    async def generate_request(_request): return JSONResponse(content={})
    async def openai_v1_chat_completions(_request): return JSONResponse(content={})
    async def openai_v1_completions(_request): return JSONResponse(content={})
    async def available_models(_request): return JSONResponse(content={"data": []})
    async def validate_json_request(_request): pass

    fake_engine_mod._launch_subprocesses = _launch_subprocesses
    fake_engine_mod.init_tokenizer_manager = init_tokenizer_manager
    fake_engine_mod.run_detokenizer_process = run_detokenizer_process
    fake_engine_mod.run_scheduler_process = run_scheduler_process
    fake_server_args_mod.prepare_server_args = prepare_server_args
    fake_http_server._GlobalState = _GlobalState
    fake_http_server.set_global_state = set_global_state
    fake_http_server.generate_request = generate_request
    fake_http_server.openai_v1_chat_completions = openai_v1_chat_completions
    fake_http_server.openai_v1_completions = openai_v1_completions
    fake_http_server.available_models = available_models
    fake_http_server.validate_json_request = validate_json_request
    fake_serving_chat.OpenAIServingChat = lambda *a, **kw: object()
    fake_serving_completions.OpenAIServingCompletion = lambda *a, **kw: object()
    fake_serving_embedding.OpenAIServingEmbedding = lambda *a, **kw: object()
    fake_serving_score.OpenAIServingScore = lambda *a, **kw: object()
    fake_serving_rerank.OpenAIServingRerank = lambda *a, **kw: object()
    fake_func_timer.enable_func_timer = lambda: None
    fake_version.__version__ = "0.0.0"
    fake_utils_mod.add_api_key_middleware = None
    fake_utils_mod.add_prometheus_middleware = None

    for name, mod in [
        ("sglang", fake_sglang),
        ("sglang.srt", fake_srt),
        ("sglang.srt.entrypoints", fake_entrypoints),
        ("sglang.srt.entrypoints.engine", fake_engine_mod),
        ("sglang.srt.entrypoints.http_server", fake_http_server),
        ("sglang.srt.server_args", fake_server_args_mod),
        ("sglang.srt.entrypoints.openai", fake_openai_mod),
        ("sglang.srt.entrypoints.openai.serving_chat", fake_serving_chat),
        ("sglang.srt.entrypoints.openai.serving_completions", fake_serving_completions),
        ("sglang.srt.entrypoints.openai.serving_embedding", fake_serving_embedding),
        ("sglang.srt.entrypoints.openai.serving_score", fake_serving_score),
        ("sglang.srt.entrypoints.openai.serving_rerank", fake_serving_rerank),
        ("sglang.srt.metrics", fake_metrics),
        ("sglang.srt.metrics.func_timer", fake_func_timer),
        ("sglang.version", fake_version),
        ("sglang.srt.utils", fake_utils_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    app = FastAPI()
    app = await init_sglang_engine(app, get_settings(), "model_path", "model_name", "info")
    paths = {route.path for route in app.router.routes}
    assert "/info" in paths
    assert "/generate" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
    assert "/v1/models" in paths
    assert "/v1/version" in paths
    with TestClient(app) as client:
        response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == app_version
    assert data["model_description"] == "model_name"
    assert data["model_type"] == ModelType.HUGGINGFACE_LLM.value
    assert data["model_card"] == {"name_or_path": "model_path"}
