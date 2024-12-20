import base64
import hashlib
import json
import logging
import re
from functools import lru_cache
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from fastapi import FastAPI, Request
from fastapi_users.jwt import decode_jwt
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware, SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from config import Settings
from exception import (
    AnnotationException,
    ClientException,
    ConfigurationException,
    StartTrainingException,
)

logger = logging.getLogger("cms")


def add_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(json.decoder.JSONDecodeError)
    async def json_decoding_exception_handler(
        _: Request, exception: json.decoder.JSONDecodeError
    ) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(_: Request, exception: RateLimitExceeded) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            content={"message": "Too many requests. Please wait and try your request again."},
        )

    @app.exception_handler(StartTrainingException)
    async def start_training_exception_handler(
        _: Request, exception: StartTrainingException
    ) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)}
        )

    @app.exception_handler(AnnotationException)
    async def annotation_exception_handler(
        _: Request, exception: AnnotationException
    ) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.exception_handler(ConfigurationException)
    async def configuration_exception_handler(
        _: Request, exception: ConfigurationException
    ) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)}
        )

    @app.exception_handler(ClientException)
    async def client_exception_handler(_: Request, exception: ClientException) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exception: Exception) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)}
        )


def add_rate_limiter(app: FastAPI, config: Settings, streamable: bool = False) -> None:
    app.state.limiter = get_rate_limiter(config)
    app.add_middleware(SlowAPIMiddleware if not streamable else SlowAPIASGIMiddleware)


@lru_cache
def get_rate_limiter(config: Settings, auth_user_enabled: Optional[bool] = None) -> Limiter:
    def get_user_auth(request: Request) -> str:
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

    auth_user_enabled = (
        config.AUTH_USER_ENABLED == "true" if auth_user_enabled is None else auth_user_enabled
    )
    return (
        Limiter(key_func=get_user_auth, strategy="moving-window")
        if auth_user_enabled
        else Limiter(key_func=get_remote_address, strategy="moving-window")
    )


def adjust_rate_limit_str(rate_limit: str) -> str:
    if "per" in rate_limit:
        return f"{int(rate_limit.split('per')[0]) * 2} per {rate_limit.split('per')[1]}"
    else:
        return f"{int(rate_limit.split('/')[0]) * 2}/{rate_limit.split('/')[1]}"


def encrypt(raw: str, public_key_pem: str) -> str:
    public_key = serialization.load_pem_public_key(public_key_pem.encode(), backend=default_backend)
    encrypted = public_key.encrypt(
        raw.encode(),  # type: ignore
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    return base64.b64encode(encrypted).decode()


def decrypt(b64_encoded: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    decrypted = private_key.decrypt(
        base64.b64decode(b64_encoded),  # type: ignore
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    return decrypted.decode()
