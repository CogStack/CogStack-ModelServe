import uuid
import logging
from typing import Optional, AsyncGenerator, List, Callable
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.authentication import AuthenticationBackend
from api.auth.db import User, get_user_db
from api.auth.backends import get_backends
from utils import get_settings

logger = logging.getLogger(__name__)


class _UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = get_settings().AUTH_JWT_SECRET
    verification_token_secret = get_settings().AUTH_JWT_SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None) -> None:
        logger.info(f"User {user.id} has registered.")

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None) -> None:
        logger.info(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None) -> None:
        logger.info(f"Verification requested for user {user.id}. Verification token: {token}")


class Props(object):

    def __init__(self, auth_user_enabled: bool) -> None:
        self._auth_backends: List = []
        self._fastapi_users = None
        self._current_active_user = lambda: None
        if auth_user_enabled:
            self._auth_backends = get_backends()
            self._fastapi_users = FastAPIUsers[User, uuid.UUID](self._get_user_manager, self.auth_backends)
            self._current_active_user = self._fastapi_users.current_user(active=True)

    @property
    def auth_backends(self) -> List[AuthenticationBackend]:
        return self._auth_backends

    @property
    def fastapi_users(self) -> Optional[FastAPIUsers]:
        return self._fastapi_users

    @property
    def current_active_user(self) -> Callable:
        return self._current_active_user

    @staticmethod
    async def _get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)) -> AsyncGenerator:
        yield _UserManager(user_db)


props = Props(get_settings().AUTH_USER_ENABLED == "true")
