import uuid
import logging
from typing import Optional, AsyncGenerator, List, Callable
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.authentication import AuthenticationBackend
from app.api.auth.db import User, get_user_db
from app.api.auth.backends import get_backends
from app.utils import get_settings

logger = logging.getLogger("cms")


class CmsUserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """
    Custom user manager class for CMS.

    Attributes:
        reset_password_token_secret (str): Secret token for resetting passwords.
        verification_token_secret (str): Secret token for verifying user accounts.
    """

    reset_password_token_secret = get_settings().AUTH_JWT_SECRET
    verification_token_secret = get_settings().AUTH_JWT_SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None) -> None:
        """
        Actions taken after a user registers.

        Args:
            user (User): The user who has registered.
            request (Optional[Request]): The optional request object.
        """

        logger.info("User %s has registered.", user.id)

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None) -> None:
        """
        Actions taken after a user requests a password reset.

        Args:
            user (User): The user who has forgotten the password.
            token (str): The reset token generated for the user.
            request (Optional[Request]): The optional request object.
        """

        logger.info("User %s has forgot their password. Reset token: %s", user.id, token)

    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None) -> None:
        """
        Actions taken after a user requests the account verification.

        Args:
            user (User): The user who has requested the verification.
            token (str): The verification token generated for the user.
            request (Optional[Request]): The optional request object.
        """

        logger.info("Verification requested for user %s. Verification token: %s", user.id, token)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)) -> AsyncGenerator:
    """
    Asynchronously generates a CMS user manager instance.

    Args:
        user_db (SQLAlchemyUserDatabase): A user database instance.

    Yields:
        CmsUserManager: A CMS user manager instance.
    """

    yield CmsUserManager(user_db)


class Props(object):
    """
    Properties class to manage authentication backends, the user instance, and the current active user.
    """

    def __init__(self, auth_user_enabled: bool) -> None:
        """
        Initialises the properties object.

        Args:
            auth_user_enabled (bool): Flag indicating whether authentication is enabled.
        """

        self._auth_backends: List = []
        self._fastapi_users = None
        self._current_active_user = lambda: None
        if auth_user_enabled:
            self._auth_backends = get_backends()
            self._fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, self.auth_backends)
            self._current_active_user = self._fastapi_users.current_user(active=True)

    @property
    def auth_backends(self) -> List[AuthenticationBackend]:
        """
        Retrieves the list of authentication backends.

        Returns:
            List[AuthenticationBackend]: A list of authentication backends.
        """

        return self._auth_backends

    @property
    def fastapi_users(self) -> Optional[FastAPIUsers]:
        """
        Retrieves the FastAPIUsers instance.

        Returns:
            Optional[FastAPIUsers]: An optional FastAPIUsers instance.
        """
        return self._fastapi_users

    @property
    def current_active_user(self) -> Callable:
        """
        Retrieves the current active user.

        Returns:
            Callable: A function to get the current active user.
        """

        return self._current_active_user
