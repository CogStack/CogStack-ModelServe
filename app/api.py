import asyncio
import globals
import importlib

from typing import Dict, Callable, Any, Optional
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from starlette.datastructures import QueryParams
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from auth.db import make_sure_db_and_tables
from domain import Tags
from management.tracker_client import TrackerClient
from dependencies import ModelServiceDep
from utils import get_settings, get_rate_limiter


def get_model_server(msd_overwritten: Optional[ModelServiceDep] = None) -> FastAPI:
    tags_metadata = [{"name": tag.name, "description": tag.value} for tag in Tags]
    app = FastAPI(openapi_tags=tags_metadata)
    app.state.limiter = get_rate_limiter()
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    if msd_overwritten is not None:
        globals.model_service_dep = msd_overwritten

    @app.on_event("startup")
    async def on_startup():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=50))
        RunVar("_default_thread_limiter").set(CapacityLimiter(50))
        (
            Instrumentator(excluded_handlers=["/docs", "/metrics", "/openapi.json", "/favicon.ico", "none"])
            .instrument(app)
            .expose(app, include_in_schema=False, should_gzip=False)
        )
        if get_settings().AUTH_USER_ENABLED == "true":
            await make_sure_db_and_tables()

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
            description="by CogStack ModelServe, a model serving and governance system for CogStack NLP solutions.",
            routes=app.routes
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://avatars.githubusercontent.com/u/28688163?s=200&v=4"
        }
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    if get_settings().AUTH_USER_ENABLED == "true":
        app = _load_auth_router(app)

    app = _load_static_router(app)
    app = _load_invocation_router(app)
    if get_settings().ENABLE_TRAINING_APIS == "true":
        app = _load_supervised_training_router(app)
        if get_settings().DISABLE_UNSUPERVISED_TRAINING != "true":
            app = _load_unsupervised_training_router(app)
        if get_settings().DISABLE_METACAT_TRAINING != "true":
            app = _load_metacat_training_router(app)

    if get_settings().ENABLE_EVALUATION_APIS == "true":
        app = _load_evaluation_router(app)
    if get_settings().ENABLE_PREVIEWS_APIS == "true":
        app = _load_preview_router(app)

    app.openapi = custom_openapi  # type: ignore

    return app


def _load_auth_router(app: FastAPI) -> FastAPI:
    from routers import authentication
    importlib.reload(authentication)
    app.include_router(authentication.router)
    return app


def _load_static_router(app: FastAPI) -> FastAPI:
    from routers import static
    importlib.reload(static)
    app.include_router(static.router)
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


def _load_metacat_training_router(app: FastAPI) -> FastAPI:
    from routers import metacat_training
    importlib.reload(metacat_training)
    app.include_router(metacat_training.router)
    return app
