"""
This module sets up the auth route for each authentication backend.
"""

import logging
import app.api.globals as cms_globals
from fastapi import APIRouter
from app.domain import Tags
router = APIRouter()
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.props.fastapi_users is not None, "FastAPI Users dependency not injected"

for auth_backend in cms_globals.props.auth_backends:
    router.include_router(
        cms_globals.props.fastapi_users.get_auth_router(auth_backend),
        prefix=f"/auth/{auth_backend.name}",
        tags=[Tags.Authentication.name],
        include_in_schema=True,
    )
