import os
import asyncio
import globals
import importlib

from functools import lru_cache
from typing import Dict, Callable, Any, Optional
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from starlette.datastructures import QueryParams
from starlette_exporter import PrometheusMiddleware, handle_metrics
from starlette_exporter.optional_metrics import response_body_size, request_body_size
from domain import Tags
from management.tracker_client import TrackerClient
from dependencies import ModelServiceDep
from config import Settings


@lru_cache()
def get_settings():
    return Settings()


def get_model_server(msd_overwritten: Optional[ModelServiceDep] = None) -> FastAPI:
    tags_metadata = [{"name": tag.name, "description": tag.value} for tag in Tags]
    app = FastAPI(openapi_tags=tags_metadata)

    if msd_overwritten is not None:
        globals.model_service_dep = msd_overwritten

    @app.on_event("startup")
    def on_startup():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=50))
        RunVar("_default_thread_limiter").set(CapacityLimiter(50))

    @app.middleware("http")
    async def verify_blank_query_params(request: Request, call_next: Callable) -> Response:
        scope = request.scope
        if request.method != "POST":
            return await call_next(request)
        if not scope or not scope.get("query_string"):
            return await call_next(request)

        query_params = QueryParams(scope["query_string"])

        scope["query_string"] = urlencode([(k, v) for k, v in query_params._list if v and v.strip()]).encode("latin-1")
        return await call_next(Request(scope, request.receive, request._send))

    @app.on_event("shutdown")
    async def on_shutdown():
        TrackerClient.end_with_interruption()

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=f"{globals.model_service_dep().model_name} APIs",
            version=globals.model_service_dep().api_version,
            description="by CogStack ModelServe, a model serving system for CogStack NLP solutions.",
            routes=app.routes
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://avatars.githubusercontent.com/u/28688163?s=200&v=4"
        }
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app = _load_metrics_route(app, globals.model_service_dep().model_name)
    app = _load_invocation_router(app)

    if get_settings().ENABLE_TRAINING_APIS == "true":
        app = _load_supervised_training_router(app)
        if get_settings().DISABLE_UNSUPERVISED_TRAINING != "true":
            app = _load_unsupervised_training_router(app)
    if get_settings().ENABLE_EVALUATION_APIS == "true":
        app = _load_evaluation_router(app)
    if get_settings().ENABLE_PREVIEWS_APIS == "true":
        app = _load_preview_router(app)

    app.openapi = custom_openapi  # type: ignore

    return app


def _load_invocation_router(app: FastAPI) -> FastAPI:
    from routers import invocation
    importlib.reload(invocation)
    app.include_router(invocation.router)
    return app


def _load_supervised_training_router(app: FastAPI) -> FastAPI:
    from routers import supervised_training
    importlib.reload(supervised_training)
    app.include_router(supervised_training.router)
    return app


def _load_evaluation_router(app: FastAPI) -> FastAPI:
    from routers import evaluation
    importlib.reload(evaluation)
    app.include_router(evaluation.router)
    return app


def _load_preview_router(app: FastAPI) -> FastAPI:
    from routers import preview
    importlib.reload(preview)
    app.include_router(preview.router)
    return app


def _load_unsupervised_training_router(app: FastAPI) -> FastAPI:
    from routers import unsupervised_training
    importlib.reload(unsupervised_training)
    app.include_router(unsupervised_training.router)
    return app


def _load_metrics_route(app: FastAPI, app_name: str) -> FastAPI:
    app.add_middleware(PrometheusMiddleware,
                       app_name=app_name,
                       prefix="cms",
                       labels={
                           "server_name": os.getenv("HOSTNAME", ""),
                           "cms_model_name": os.getenv("CMS_MODEL_NAME", ""),
                       },
                       buckets=[0.1, 0.25, 0.5, 1],
                       skip_paths=["/", "/docs", "/favicon.ico", "/metrics", "/openapi.json"],
                       optional_metrics=[request_body_size, response_body_size])
    app.add_route("/metrics", handle_metrics)
    return app
