import asyncio
import importlib
import logging
import os.path
import api.globals as globals

from typing import Dict, Callable, Any, Optional
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from starlette.datastructures import QueryParams
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_400_BAD_REQUEST
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.auth.db import make_sure_db_and_tables
from domain import Tags
from management.tracker_client import TrackerClient
from api.dependencies import ModelServiceDep
from exception import StartTrainingException, AnnotationException
from utils import get_settings, get_rate_limiter, rate_limit_exceeded_handler


logger = logging.getLogger(__name__)


def get_model_server(msd_overwritten: Optional[ModelServiceDep] = None) -> FastAPI:
    tags_metadata = [{"name": tag.name, "description": tag.value} for tag in Tags]
    app = FastAPI(title="CogStack ModelServe",
                  summary="A model serving and governance system for CogStack NLP solutions",
                  docs_url=None,
                  redoc_url=None,
                  debug=(get_settings().DEBUG == "true"),
                  openapi_tags=tags_metadata)
    app.state.limiter = get_rate_limiter()
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    instrumentator = Instrumentator(excluded_handlers=["/docs", "/redoc", "/metrics", "/openapi.json", "/favicon.ico", "none"]).instrument(app)

    if msd_overwritten is not None:
        globals.model_service_dep = msd_overwritten

    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

    @app.on_event("startup")
    async def on_startup() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=50))
        RunVar("_default_thread_limiter").set(CapacityLimiter(50))  # type: ignore
        instrumentator.expose(app, include_in_schema=False, should_gzip=False)
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

    @app.get("/docs", include_in_schema=False)
    async def swagger_doc(req: Request) -> HTMLResponse:
        root_path = req.scope.get("root_path", "").rstrip("/")
        openapi_url = root_path + app.openapi_url
        oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url
        if oauth2_redirect_url:
            oauth2_redirect_url = root_path + oauth2_redirect_url
        return get_swagger_ui_html(
            openapi_url=openapi_url,
            title="CogStack ModelServe",
            oauth2_redirect_url=oauth2_redirect_url,
            init_oauth=app.swagger_ui_init_oauth,
            swagger_favicon_url="/static/images/favicon.ico",
            swagger_ui_parameters=app.swagger_ui_parameters,
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_doc(req: Request) -> HTMLResponse:
        root_path = req.scope.get("root_path", "").rstrip("/")
        openapi_url = root_path + app.openapi_url
        return get_redoc_html(
            openapi_url=openapi_url,
            title="CogStack ModelServe",
            redoc_favicon_url="/static/images/favicon.ico",
        )

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.exception_handler(StartTrainingException)
    async def start_training_exception_handler(_: Request, exception: StartTrainingException) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"message": str(exception)})

    @app.exception_handler(AnnotationException)
    async def annotation_exception_handler(_: Request, exception: AnnotationException) -> JSONResponse:
        logger.exception(exception)
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": str(exception)})

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
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
        for path in openapi_schema["paths"].values():
            for method_data in path.values():
                if "requestBody" in method_data:
                    for content_type, content in method_data["requestBody"]["content"].items():
                        if content_type == "multipart/form-data":
                            schema_name = content["schema"]["$ref"].lstrip("#/components/schemas/")
                            schema_data = openapi_schema["components"]["schemas"].pop(schema_name)
                            schema_data["title"] = "UploadFile"
                            content["schema"] = schema_data
                        elif content_type == "application/x-www-form-urlencoded":
                            schema_name = content["schema"]["$ref"].lstrip("#/components/schemas/")
                            schema_data = openapi_schema["components"]["schemas"].pop(schema_name)
                            schema_data["title"] = "FormData"
                            content["schema"] = schema_data
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    if get_settings().AUTH_USER_ENABLED == "true":
        app = _load_auth_router(app)

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
    from api.routers import authentication
    importlib.reload(authentication)
    app.include_router(authentication.router)
    return app


def _load_invocation_router(app: FastAPI) -> FastAPI:
    from api.routers import invocation
    importlib.reload(invocation)
    app.include_router(invocation.router)
    return app


def _load_supervised_training_router(app: FastAPI) -> FastAPI:
    from api.routers import supervised_training
    importlib.reload(supervised_training)
    app.include_router(supervised_training.router)
    return app


def _load_evaluation_router(app: FastAPI) -> FastAPI:
    from api.routers import evaluation
    importlib.reload(evaluation)
    app.include_router(evaluation.router)
    return app


def _load_preview_router(app: FastAPI) -> FastAPI:
    from api.routers import preview
    importlib.reload(preview)
    app.include_router(preview.router)
    return app


def _load_unsupervised_training_router(app: FastAPI) -> FastAPI:
    from api.routers import unsupervised_training
    importlib.reload(unsupervised_training)
    app.include_router(unsupervised_training.router)
    return app


def _load_metacat_training_router(app: FastAPI) -> FastAPI:
    from api.routers import metacat_training
    importlib.reload(metacat_training)
    app.include_router(metacat_training.router)
    return app
