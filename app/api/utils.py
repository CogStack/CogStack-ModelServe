import json
import logging
import re
import hashlib
import base64
import contextlib
import uuid
import tempfile
from functools import lru_cache
from typing import Optional, AsyncGenerator, Dict, Any
from typing_extensions import Annotated
from fastapi import FastAPI, Request, APIRouter, Body, Query
from starlette.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send
from starlette.status import (
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_501_NOT_IMPLEMENTED,
    HTTP_400_BAD_REQUEST,
    HTTP_429_TOO_MANY_REQUESTS,
)
from slowapi.middleware import SlowAPIMiddleware, SlowAPIASGIMiddleware
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi_users.jwt import decode_jwt
from app import __version__ as app_version
from app.config import Settings
from app.domain import TagsGenerative, ModelCard, ModelType
from app.processors.prompt_factory import PromptFactory
from app.exception import (
    StartTrainingException,
    AnnotationException,
    ConfigurationException,
    ClientException,
    ExtraDependencyRequiredException,
    GenerationException,
)
from app.utils import get_settings, has_turing_generation_gpu

logger = logging.getLogger("cms")


class ForwardedPrefixMiddleware:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            forwarded_prefix = next(
                (
                    v.decode("latin-1").strip()
                    for k, v in scope.get("headers", [])
                    if k == b"x-forwarded-prefix"
                ),
                None,
            )
            if forwarded_prefix:
                scope["root_path"] = "/" + forwarded_prefix.strip("/")
        await self.app(scope, receive, send)


def add_exception_handlers(app: FastAPI) -> None:
    """
    Adds custom exception handlers to the FastAPI app instance.

    Args:
        app (FastAPI): The FastAPI app instance.
    """

    @app.exception_handler(json.decoder.JSONDecodeError)
    async def json_decoding_exception_handler(_: Request, exception: json.decoder.JSONDecodeError) -> JSONResponse:
        """
        Handles JSON decoding errors.

        Args:
           _ (Request): The request object.
           exception (JSONDecodeError): The JSON decoding error.

        Returns:
           JSONResponse: A JSON response with a 400 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(_: Request, exception: RateLimitExceeded) -> JSONResponse:
        """
        Handles rate limit exceeded exceptions.

        Args:
            _ (Request): The request object.
            exception (RateLimitExceeded): The rate limit exceeded exception.

        Returns:
            JSONResponse: A JSON response with a 429 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            content={"message": "Too many requests. Please wait and try your request again."},
        )

    @app.exception_handler(StartTrainingException)
    async def start_training_exception_handler(_: Request, exception: StartTrainingException) -> JSONResponse:
        """
        Handles start training exceptions.

        Args:
            _ (Request): The request object.
            exception (StartTrainingException): The start training exception.

        Returns:
            JSONResponse: A JSON response with a 500 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)})

    @app.exception_handler(AnnotationException)
    async def annotation_exception_handler(_: Request, exception: AnnotationException) -> JSONResponse:
        """
        Handles annotation exceptions.

        Args:
            _ (Request): The request object.
            exception (AnnotationException): The annotation exception.

        Returns:
            JSONResponse: A JSON response with a 400 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.exception_handler(ConfigurationException)
    async def configuration_exception_handler(_: Request, exception: ConfigurationException) -> JSONResponse:
        """
        Handles configuration exceptions.

        Args:
            _ (Request): The request object.
            exception (ConfigurationException): The configuration exception.

        Returns:
            JSONResponse: A JSON response with a 500 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)})

    @app.exception_handler(ExtraDependencyRequiredException)
    async def extra_dependency_exception_handler(
        _: Request,
        exception: ExtraDependencyRequiredException
    ) -> JSONResponse:
        """
        Handles extra dependency required exceptions.

        Args:
            _ (Request): The request object.
            exception (ExtraDependencyRequiredException): The extra dependency required exception.

        Returns:
            JSONResponse: A JSON response with a 500 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)})

    @app.exception_handler(ClientException)
    async def client_exception_handler(_: Request, exception: ClientException) -> JSONResponse:
        """
        Handles client exceptions.

        Args:
            _ (Request): The request object.
            exception (ClientException): The client exception.

        Returns:
            JSONResponse: A JSON response with a 400 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.exception_handler(GenerationException)
    async def generation_exception_handler(_: Request, exception: GenerationException) -> JSONResponse:
        """
        Handles generation exceptions.

        Args:
            _ (Request): The request object.
            exception (GenerationException): The generation exception.

        Returns:
            JSONResponse: A JSON response with a 500 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exception: Exception) -> JSONResponse:
        """
        Handles all other exceptions.

        Args:
            _ (Request): The request object.
            exception (Exception): The unhandled exception.

        Returns:
            JSONResponse: A JSON response with a 500 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)})

    @app.exception_handler(NotImplementedError)
    async def not_implemented_exception_handler(_: Request, exception: NotImplementedError) -> JSONResponse:
        """
        Handles not implemented exceptions.

        Args:
            _ (Request): The request object.
            exception (NotImplementedError): The not implemented exception.

        Returns:
            JSONResponse: A JSON response with a 501 status code and an error message.
        """
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_501_NOT_IMPLEMENTED, content={"message": str(exception)})


def add_rate_limiter(app: FastAPI, config: Settings, streamable: bool = False) -> None:
    """
    Adds a rate limiter to the FastAPI app instance.

    Args:
        app (FastAPI): The FastAPI app instance.
        config (Settings): Configuration settings for the model service.
        streamable (bool): Whether the app is streamable or not. Defaults to False.
    """
    app.state.limiter = get_rate_limiter(config)
    app.add_middleware(SlowAPIMiddleware if not streamable else SlowAPIASGIMiddleware)


@lru_cache
def get_rate_limiter(config: Settings, auth_user_enabled: Optional[bool] = None) -> Limiter:
    """
    Retrieves a rate limiter based on the app configuration.

    Args:
        config (Settings): Configuration settings for the model service.
        auth_user_enabled (Optional[bool]): Whether to use user auth as the limit key or not. If None, remote address is used.

    Returns:
        Limiter: A rate limiter configured to use either user auth or remote address as the limit key.
    """

    def _get_user_auth(request: Request) -> str:
        request_headers = request.scope.get("headers", [])
        limiter_prefix = request.scope.get("root_path", "") + request.scope.get("path") + ":"
        current_key = ""

        for headers in request_headers:
            if headers[0].decode() == "authorization":
                token = headers[1].decode().split("Bearer ")[1]
                payload = decode_jwt(token, config.AUTH_JWT_SECRET, ["fastapi-users:auth"])
                sub = payload.get("sub")
                assert sub is not None, "Cannot find 'sub' in the decoded payload"
                hash_object = hashlib.sha256(sub.encode())
                current_key = hash_object.hexdigest()
                break

        limiter_key = re.sub(r":+", ":", re.sub(r"/+", ":", limiter_prefix + current_key))
        return limiter_key

    auth_user_enabled = config.AUTH_USER_ENABLED == "true" if auth_user_enabled is None else auth_user_enabled
    if auth_user_enabled:
        return Limiter(key_func=_get_user_auth, strategy="moving-window")
    else:
        return Limiter(key_func=get_remote_address, strategy="moving-window")


def adjust_rate_limit_str(rate_limit: str) -> str:
    """
    Adjusts the rate limit string.

    Args:
        rate_limit (str): The original rate limit string in the format 'X per Y' or 'X/Y'.

    Returns:
        str: The adjusted rate limit string.
    """

    if "per" in rate_limit:
        return f"{int(rate_limit.split('per')[0]) * 2} per {rate_limit.split('per')[1]}"
    else:
        return f"{int(rate_limit.split('/')[0]) * 2}/{rate_limit.split('/')[1]}"


def encrypt(raw: str, public_key_pem: str) -> str:
    """
    Encrypts a raw string using a public key.

    Args:
        raw (str): The raw string to be encrypted.
        public_key_pem (str): The public key in the PEM format.

    Returns:
        str: The encrypted string.
    """

    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend)
    encrypted = public_key.encrypt(    # type: ignore
        raw.encode(),
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return base64.b64encode(encrypted).decode()


def decrypt(b64_encoded: str, private_key_pem: str) -> str:
    """
    Decrypts a base64 encoded string using a private key.

    Args:
        b64_encoded (str): The base64 encoded encrypted string.
        private_key_pem (str): The private key in the PEM format.

    Returns:
        str: The decrypted string.
    """

    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    decrypted = private_key.decrypt(    # type: ignore
        base64.b64decode(b64_encoded),
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return decrypted.decode()


async def init_vllm_engine(
    app: FastAPI,
    config: Settings,
    model_dir_path: str,
    model_name: str,
    log_level: str = "info",
    server_args: Optional[str] = None,
) -> FastAPI:
    """
    Initialises the vLLM engine.

    Args:
        app (FastAPI): The FastAPI app instance.
        config (Settings): Configuration settings for the model service.
        model_dir_path (str): The path to the directory containing the model.
        model_name (str): The name of the model.
        log_level (str): The log level for the VLLM engine. Defaults to "info".
        server_args (Optional[str]): The arguments to pass to the vLLM engine.
    """

    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
        from vllm.entrypoints.chat_utils import parse_chat_messages, apply_hf_chat_template
        from vllm.entrypoints.openai.api_server import (
            create_chat_completion,
            create_completion,
            show_available_models,
            show_version,
            build_async_engine_client_from_engine_args,
            init_app_state,
        )
        from vllm import SamplingParams, TokensPrompt
    except ImportError:
        logger.error("Cannot import the vLLM engine. Please install it with `pip install '.[vllm]'`.")
        raise ExtraDependencyRequiredException("Cannot import the vLLM engine. Please install it with `pip install '.[vllm]'`.")

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args(server_args.split() if server_args else [])
    validate_parsed_serve_args(args)

    args.model = model_dir_path
    args.dtype = "float16"
    args.served_model_name = [model_name]
    args.max_model_len = 2048
    args.uvicorn_log_level = log_level
    if hasattr(args, "chat_template") and config.OVERRIDE_CHAT_TEMPLATE:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False)
        tmp.write(config.OVERRIDE_CHAT_TEMPLATE)
        tmp.flush()
        args.chat_template = tmp.name
    if hasattr(args, "enable_auto_tool_choice"):
        args.enable_auto_tool_choice = True
    if hasattr(args, "tool_call_parser"):
        args.tool_call_parser = "pythonic"
    if hasattr(args, "return_tokens_as_token_ids"):
        args.return_tokens_as_token_ids = True
    if hasattr(args, "default_chat_template_kwargs"):
        args.default_chat_template_kwargs = {"enable_thinking": False}

    exit_stack = contextlib.AsyncExitStack()
    engine = await exit_stack.enter_async_context(
        build_async_engine_client_from_engine_args(
            AsyncEngineArgs.from_cli_args(args),
            disable_frontend_multiprocessing=True,
        )
    )
    app.state._vllm_exit_stack = exit_stack
    app.state._vllm_engine = engine

    tokenizer = await engine.get_tokenizer()
    model_config = getattr(engine, "model_config", None)
    if model_config is None:
        vllm_config = getattr(engine, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
    await init_app_state(engine, app.state, args)

    async def get_model_card() -> ModelCard:
        return ModelCard(
            model_description=model_name,
            model_type=ModelType.HUGGINGFACE_LLM,
            api_version=app_version,
            model_card=_to_model_card_dict(model_config),
        )

    async def generate_text(
        request: Request,
        prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
        max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512
    ) -> StreamingResponse:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        params = SamplingParams(max_tokens=max_tokens)

        conversation, _, _ = parse_chat_messages(  # type: ignore
            messages=messages,  # type: ignore[arg-type]
            model_config=model_config,  # type: ignore[arg-type]
            content_format="string",    # type: ignore[arg-type]
        )
        chat_template: Optional[str] = None
        if args.chat_template:
            chat_template = args.chat_template
        else:
            if getattr(tokenizer, "chat_template", None):  # type: ignore
                chat_template = tokenizer.chat_template  # type: ignore
            elif getattr(tokenizer, "default_chat_template", None):  # type: ignore
                tokenizer.chat_template = tokenizer.default_chat_template  # type: ignore
                chat_template = tokenizer.chat_template  # type: ignore
            else:
                chat_template = PromptFactory.create_chat_template()

        prompt_text = apply_hf_chat_template(
            tokenizer,
            conversation=conversation,
            tools=None,
            add_generation_prompt=True,
            continue_final_message=False,
            model_config=model_config,  # type: ignore
            chat_template=chat_template,
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_obj = TokensPrompt(prompt_token_ids=prompt_token_ids)

        async def _stream() -> AsyncGenerator[bytes, None]:
            start = 0
            async for output in engine.generate(request_id=uuid.uuid4().hex, prompt=prompt_obj, sampling_params=params):
                text = output.outputs[0].text
                yield text[start:].encode("utf-8")
                start = len(text)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    router = APIRouter()
    endpoints = [
        ["/info", get_model_card, ["GET"]],
        ["/generate", generate_text, ["POST"]],
        ["/v1/chat/completions", create_chat_completion, ["POST"]],
        ["/v1/completions", create_completion, ["POST"]],
        ["/v1/models", show_available_models, ["GET"]],
        ["/v1/version", show_version, ["GET"]],
    ]

    for route, endpoint, methods in endpoints:
        router.add_api_route(
            path=route,
            endpoint=endpoint,
            methods=methods,
            include_in_schema=True,
            tags=[TagsGenerative.Generative.name],
        )
    app.include_router(router)

    return app


async def init_sglang_engine(
    app: FastAPI,
    config: Settings,
    model_dir_path: str,
    model_name: str,
    log_level: str = "info",
    server_args: Optional[str] = None,
) -> FastAPI:
    """
    Initialises the SGLang engine.

    Args:
        app (FastAPI): The FastAPI app instance.
        config (Settings): Configuration settings for the model service.
        model_dir_path (str): The path to the directory containing the model.
        model_name (str): The name of the model.
        log_level (str): The log level for the SGLang engine. Defaults to "info".
        server_args (Optional[str]): The arguments to pass to the SGLang engine.
    """

    try:
        from sglang.srt.entrypoints.engine import (
            _launch_subprocesses,
            init_tokenizer_manager,
            run_detokenizer_process,
            run_scheduler_process,
        )
        from fastapi import Depends
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.entrypoints.http_server import (
            _GlobalState,
            set_global_state,
            generate_request,
            openai_v1_completions,
            openai_v1_chat_completions,
            available_models,
            validate_json_request,
        )
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
        from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
        from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
        from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
        from sglang.srt.metrics.func_timer import enable_func_timer
        from sglang.version import __version__ as sglang_version
    except ImportError:
        logger.error("Cannot import the SGLang engine. Please install it with `pip install '.[sglang]'`.")
        raise ExtraDependencyRequiredException("Cannot import the SGLang engine. Please install it with `pip install '.[sglang]'`.")

    add_api_key_middleware = None
    add_prometheus_middleware = None
    try:
        from sglang.srt.utils import add_api_key_middleware, add_prometheus_middleware # type: ignore
    except ImportError:
        logger.warning(
            "SGLang middleware helpers not available in this version; "
            "API-key and Prometheus middleware setup will be skipped."
        )

    server_args = prepare_server_args((server_args.split() if server_args else []) + ["--model-path", model_dir_path])
    server_args.served_model_name = model_name
    server_args.log_level = log_level
    server_args.log_level_http = log_level
    server_args.tokenizer_worker_num = 1
    server_args.skip_server_warmup = False
    server_args.quantization = None # "bitsandbytes"
    server_args.model_impl = "transformers"
    server_args.mem_fraction_static = 0.9

    if config.OVERRIDE_CHAT_TEMPLATE:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False)
        tmp.write(config.OVERRIDE_CHAT_TEMPLATE)
        tmp.flush()
        server_args.chat_template = tmp.name

    if has_turing_generation_gpu():
        server_args.sampling_backend = "pytorch"
        server_args.attention_backend = None if get_settings().ENABLE_SPDA_ATTN == "true" else "torch_native"
        server_args.prefill_attention_backend =  None if get_settings().ENABLE_SPDA_ATTN == "true" else "torch_native"
        server_args.decode_attention_backend =  None if get_settings().ENABLE_SPDA_ATTN == "true" else "torch_native"
        server_args.disable_cuda_graph = True

    result = _launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager,
        run_scheduler_process_func=run_scheduler_process,
        run_detokenizer_process_func=run_detokenizer_process,
    )

    if len(result) == 4:
        tokenizer_manager, template_manager, scheduler_infos, subprocess_watchdog = result
        if not scheduler_infos:
            raise ExtraDependencyRequiredException(
                "SGLang engine started but scheduler_infos is empty; cannot build HTTP global state."
            )
        if tokenizer_manager is not None and subprocess_watchdog is not None:
            tokenizer_manager._subprocess_watchdog = subprocess_watchdog
    else:
        raise ExtraDependencyRequiredException(
            f"Unexpected _launch_subprocesses return length {len(result)}; expected 4."
        )

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )

    if server_args.api_key and add_api_key_middleware is not None:
        add_api_key_middleware(app, server_args.api_key)
    elif server_args.api_key:
        logger.warning("SGLang API key middleware is unavailable in this version.")
    if server_args.enable_metrics and add_prometheus_middleware is not None:
        add_prometheus_middleware(app)
        enable_func_timer()
    elif server_args.enable_metrics:
        logger.warning("SGLang Prometheus middleware is unavailable in this version.")

    app.state.openai_serving_completion = OpenAIServingCompletion(tokenizer_manager, template_manager)
    app.state.openai_serving_chat = OpenAIServingChat(tokenizer_manager, template_manager)
    app.state.openai_serving_embedding = OpenAIServingEmbedding(tokenizer_manager, template_manager)
    app.state.openai_serving_score = OpenAIServingScore(tokenizer_manager)
    app.state.openai_serving_rerank = OpenAIServingRerank(tokenizer_manager)

    async def get_model_card() -> ModelCard:
        model_config = getattr(tokenizer_manager, "model_config", None)
        return ModelCard(
            model_description=model_name,
            model_type=ModelType.HUGGINGFACE_LLM,
            api_version=app_version,
            model_card=_to_model_card_dict(model_config),
        )

    async def show_version() -> Dict[str, str]:
        return {"version": sglang_version}

    router = APIRouter()
    endpoints = [
        ["/info", get_model_card, ["GET"], None],
        ["/generate", generate_request, ["POST"], None],
        ["/v1/chat/completions", openai_v1_chat_completions, ["POST"], [Depends(validate_json_request)]],
        ["/v1/completions", openai_v1_completions, ["POST"], [Depends(validate_json_request)]],
        ["/v1/models", available_models, ["GET"], None],
        ["/v1/version", show_version, ["GET"], None],
    ]

    for route, endpoint, methods, dependencies in endpoints:
        router.add_api_route(
            path=route,
            endpoint=endpoint,
            methods=methods,
            dependencies=dependencies,
            include_in_schema=True,
            tags=[TagsGenerative.Generative.name],
        )
    app.include_router(router)

    return app


def _to_model_card_dict(model_config: Optional[Any]) -> Dict:
    """
    Converts a model config object into a serialisable dict when possible.
    """

    if model_config is None:
        return {}
    if hasattr(model_config, "to_dict"):
        try:
            return model_config.to_dict()  # type: ignore[no-any-return]
        except Exception:
            logger.exception("Failed to convert model config with to_dict().")
    if isinstance(model_config, dict):
        return model_config
    if hasattr(model_config, "__dict__"):
        try:
            return dict(vars(model_config))
        except Exception:
            logger.exception("Failed to convert model config using vars().")
    return {}
