from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from api.auth.users import props


router = APIRouter()


@router.get("/healthz",
            dependencies=[Depends(props.current_active_user)],
            description="Health check endpoint",
            include_in_schema=False)
async def is_healthy() -> PlainTextResponse:
    return PlainTextResponse(content="OK", status_code=200)
