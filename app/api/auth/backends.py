from functools import lru_cache
from typing import List
from fastapi_users.authentication.transport.base import Transport
from fastapi_users.authentication.strategy.base import Strategy
from fastapi_users.authentication import BearerTransport, JWTStrategy
from fastapi_users.authentication import AuthenticationBackend, CookieTransport
from app.utils import get_settings


@lru_cache
def get_backends() -> List[AuthenticationBackend]:
    """
    Retrieves a list of authentication backends used when CMS APIs are invoked.

    Returns:
        List[AuthenticationBackend]: A list of authentication backends.
    """

    return [
        AuthenticationBackend(name="jwt", transport=_get_bearer_transport(), get_strategy=_get_strategy),
        AuthenticationBackend(name="cookie", transport=_get_cookie_transport(), get_strategy=_get_strategy),
    ]


def _get_bearer_transport() -> Transport:
    return BearerTransport(tokenUrl="auth/jwt/login")


def _get_cookie_transport() -> Transport:
    return CookieTransport(cookie_max_age=get_settings().AUTH_ACCESS_TOKEN_EXPIRE_SECONDS)


def _get_strategy() -> Strategy:
    return JWTStrategy(secret=get_settings().AUTH_JWT_SECRET, lifetime_seconds=get_settings().AUTH_ACCESS_TOKEN_EXPIRE_SECONDS)
