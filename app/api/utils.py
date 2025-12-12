import json
import logging
import re
import hashlib
import base64
import contextlib
import uuid
from functools import lru_cache
from typing import Optional, AsyncGenerator
from typing_extensions import Annotated
from fastapi import FastAPI, Request, APIRouter, Body, Query
from starlette.responses import JSONResponse, StreamingResponse
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
from app.config import Settings
from app.domain import TagsGenerative
from app.exception import (
    StartTrainingException,
    AnnotationException,
    ConfigurationException,
    ClientException,
    ExtraDependencyRequiredException,
)

logger = logging.getLogger("cms")


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

async def init_vllm_engine(app: FastAPI,
                           model_dir_path: str,
                           model_name: str,
                           log_level: str = "info") -> FastAPI:
    """
    Initialises the vLLM engine.

    Args:
        app (FastAPI): The FastAPI app instance.
        model_dir_path (str): The path to the directory containing the model.
        model_name (str): The name of the model.
        log_level (str): The log level for the VLLM engine. Defaults to "info".
    """

    try:
        # Import necessary vLLM components
        from vllm.utils import FlexibleArgumentParser
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
        from vllm.entrypoints.chat_utils import parse_chat_messages, apply_hf_chat_template
        from vllm.entrypoints.openai.api_server import (
            create_chat_completion,
            show_available_models,
            build_async_engine_client_from_engine_args,
            init_app_state,
        )
        from vllm import SamplingParams, TokensPrompt
    except ImportError:
        logger.error("Cannot import the vLLM engine. Please install it with `pip install cms[llm]`.")
        raise ExtraDependencyRequiredException("Cannot import the vLLM engine. Please install it with `pip install cms[llm]`.")

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args([])
    validate_parsed_serve_args(args)

    args.model = model_dir_path
    args.dtype = "float16"
    args.served_model_name = [model_name]
    args.max_model_len = 2048 # The default batched length (2048) needs to be higher than max_model_len.
    # args.tokenizer = model_dir_path # Uncomment if your tokenizer is in a different path or needs explicit setting.
    args.log_level = log_level

    exit_stack = contextlib.AsyncExitStack()
    engine = await exit_stack.enter_async_context(
        build_async_engine_client_from_engine_args(
            AsyncEngineArgs.from_cli_args(args),
            disable_frontend_multiprocessing=True,
        )
    )

    tokenizer = await engine.get_tokenizer()
    vllm_config = await engine.get_vllm_config()
    model_config = await engine.get_model_config()

    await init_app_state(engine, vllm_config, app.state, args)

    async def generate_text(
        request: Request,
        prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
        max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512
    ) -> StreamingResponse:
        """
        Custom endpoint for streaming text generation.
        This endpoint takes a raw text prompt and streams back the generated text.
        It applies a chat template to the prompt internally for model compatibility.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        params = SamplingParams(max_tokens=max_tokens)

        conversation, _ = parse_chat_messages(messages, model_config, tokenizer, content_format="string")   # type: ignore
        prompt_tokens = apply_hf_chat_template( # type: ignore
            tokenizer,
            conversation=conversation,
            tools=None,
            add_generation_prompt=True,
            continue_final_message=False,
            chat_template="{% for message in messages %}\n{% if message['role'] == 'user' %}\nUser: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}\nAssistant: {{ message['content'] }}\n{% endif %}\n{% endfor %}\nAssistant:",
            tokenize=True,
        )
        prompt_obj = TokensPrompt(prompt_token_ids=prompt_tokens)   # type: ignore

        async def _stream() -> AsyncGenerator[bytes, None]:
            start = 0
            async for output in engine.generate(request_id=uuid.uuid4().hex, prompt=prompt_obj, sampling_params=params):
                text = output.outputs[0].text
                yield text[start:].encode("utf-8")
                start = len(text)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    router = APIRouter()
    endpoints = [
        ["/generate", generate_text, ["POST"]],
        ["/chat/completions", create_chat_completion, ["POST"]],
        ["/models", show_available_models, ["GET"]],
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
