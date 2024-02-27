import api.globals as cms_globals
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from model_services.base import AbstractModelService

router = APIRouter()


@router.get("/healthz",
            description="Health check endpoint",
            include_in_schema=False)
async def is_healthy() -> PlainTextResponse:
    return PlainTextResponse(content="OK", status_code=200)


@router.get("/readyz",
            description="Readiness check endpoint",
            include_in_schema=False)
async def is_ready(model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> PlainTextResponse:
    return PlainTextResponse(content=model_service.info().model_type, status_code=200)
