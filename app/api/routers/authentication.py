import logging

from fastapi import APIRouter

from domain import Tags

import api.globals as cms_globals

router = APIRouter()
logger = logging.getLogger("cms")

for auth_backend in cms_globals.props.auth_backends:
    router.include_router(
        cms_globals.props.fastapi_users.get_auth_router(auth_backend),
        prefix=f"/auth/{auth_backend.name}",
        tags=[Tags.Authentication.name],
        include_in_schema=True,
    )
