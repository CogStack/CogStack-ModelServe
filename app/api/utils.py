import json
import logging
import re
import hashlib
import base64
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_400_BAD_REQUEST, HTTP_429_TOO_MANY_REQUESTS
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
from app.exception import StartTrainingException, AnnotationException, ConfigurationException, ClientException

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
        return JSONResponse(status_code=HTTP_429_TOO_MANY_REQUESTS, content={"message": "Too many requests. Please wait and try your request again."})

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
    return Limiter(key_func=_get_user_auth, strategy="moving-window") if auth_user_enabled else Limiter(key_func=get_remote_address, strategy="moving-window")


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
    encrypted = public_key.encrypt(raw.encode(),  # type: ignore
                                   padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
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
    decrypted = private_key.decrypt(base64.b64decode(b64_encoded),  # type: ignore
                                    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return decrypted.decode()
