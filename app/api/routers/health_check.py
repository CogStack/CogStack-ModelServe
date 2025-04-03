import app.api.globals as cms_globals
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from app.model_services.base import AbstractModelService

router = APIRouter()

assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.get(
"/healthz",
    description="Health check endpoint",
    include_in_schema=False,
)
async def is_healthy() -> PlainTextResponse:
    """
    Performs a health check to ensure the FastAPI service is in operation.

    Returns:
        PlainTextResponse: A response with content "OK" and status code 200 if the service is healthy.
    """

    return PlainTextResponse(content="OK", status_code=200)


@router.get(
    "/readyz",
    description="Readiness check endpoint",
    include_in_schema=False,
)
async def is_ready(model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> PlainTextResponse:
    """
    Performs a readiness check to ensure the model service is ready for scoring and training tasks.

    Args:
        model_service (AbstractModelService): The model service dependency.

    Returns:
        PlainTextResponse: A response with the model type as content and status code 200 if the service is ready.
    """

    return PlainTextResponse(content=model_service.info().model_type, status_code=200)
