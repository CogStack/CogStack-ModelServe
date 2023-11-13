import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/favicon.ico",
            include_in_schema=False,
            response_class=FileResponse)
async def favicon() -> FileResponse:
    return FileResponse(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "static", "images", "favicon.ico"))
