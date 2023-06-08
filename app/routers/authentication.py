import logging

from fastapi import APIRouter
from domain import Tags
from auth.users import props
router = APIRouter()
logger = logging.getLogger(__name__)

for auth_backend in props.auth_backends:
    router.include_router(
        props.fastapi_users.get_auth_router(auth_backend),
        prefix=f"/auth/{auth_backend.name}",
        tags=[Tags.Authentication.name],
        include_in_schema=True,
    )
