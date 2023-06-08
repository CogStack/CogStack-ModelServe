from functools import lru_cache
from typing import List
from fastapi_users.authentication.transport.base import Transport
from fastapi_users.authentication.strategy.base import Strategy
from fastapi_users.authentication import BearerTransport, JWTStrategy
from fastapi_users.authentication import AuthenticationBackend
from utils import get_settings


@lru_cache
def get_backends() -> List[AuthenticationBackend]:
    return [
        AuthenticationBackend(name="jwt", transport=_get_transport(), get_strategy=_get_strategy),
    ]


def _get_transport() -> Transport:
    return BearerTransport(tokenUrl="auth/jwt/login")


def _get_strategy() -> Strategy:
    return JWTStrategy(secret=get_settings().AUTH_JWT_SECRET, lifetime_seconds=get_settings().AUTH_ACCESS_TOKEN_EXPIRE_SECONDS)
