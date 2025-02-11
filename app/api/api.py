import logging
import asyncio
import importlib
import os.path
import app.api.globals as cms_globals

from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.auth.db import make_sure_db_and_tables
from app.api.auth.users import Props
from app.api.dependencies import ModelServiceDep
from app.api.utils import add_exception_handlers, add_rate_limiter
from app.domain import Tags, TagsStreamable
from app.management.tracker_client import TrackerClient
from app.utils import get_settings


logging.getLogger("asyncio").setLevel(logging.ERROR)
logger = logging.getLogger("cms")


def get_model_server(msd_overwritten: Optional[ModelServiceDep] = None) -> FastAPI:
    app = _get_app(msd_overwritten)
    config = get_settings()
    logger.debug("Configuration loaded: %s", config)
    add_rate_limiter(app, config)
    logger.debug("Rate limiter added")

    app = _load_health_check_router(app)
    logger.debug("Health check router loaded")

    if config.AUTH_USER_ENABLED == "true":
        app = _load_auth_router(app)
        logger.debug("Auth router loaded")

    app = _load_model_card(app)
    logger.debug("Model card router loaded")
    app = _load_invocation_router(app)
    logger.debug("Invocation router loaded")

    if config.ENABLE_TRAINING_APIS == "true":
        app = _load_supervised_training_router(app)
        logger.debug("Supervised training router loaded")
        if config.DISABLE_UNSUPERVISED_TRAINING != "true":
            app = _load_unsupervised_training_router(app)
            logger.debug("Unsupervised training router loaded")
        if config.DISABLE_METACAT_TRAINING != "true":
            app = _load_metacat_training_router(app)
            logger.debug("Metacat training router loaded")
        app = _load_training_operations(app)

    if config.ENABLE_EVALUATION_APIS == "true":
        app = _load_evaluation_router(app)
        logger.debug("Evaluation router loaded")
    if config.ENABLE_PREVIEWS_APIS == "true":
        app = _load_preview_router(app)
        logger.debug("Preview router loaded")

    return app


def get_stream_server(msd_overwritten: Optional[ModelServiceDep] = None) -> FastAPI:
    app = _get_app(msd_overwritten, streamable=True)
    config = get_settings()
    add_rate_limiter(app, config, streamable=True)

    app = _load_health_check_router(app)
    logger.debug("Health check router loaded")

    if config.AUTH_USER_ENABLED == "true":
        app = _load_auth_router(app)
        logger.debug("Auth router loaded")

    app = _load_model_card(app)
    logger.debug("Model card router loaded")
    app = _load_stream_router(app)
    logger.debug("Stream router loaded")

    return app


def _get_app(msd_overwritten: Optional[ModelServiceDep] = None, streamable: bool = False) -> FastAPI:
    tags_metadata = [{  # type: ignore
        "name": tag.name,
        "description": tag.value
    } for tag in (Tags if not streamable else TagsStreamable)]
    config = get_settings()
    app = FastAPI(title="CogStack ModelServe",
                  summary="A model serving and governance system for CogStack NLP solutions",
                  docs_url=None,
                  redoc_url=None,
                  debug=(config.DEBUG == "true"),
                  openapi_tags=tags_metadata)
    add_exception_handlers(app)
    instrumentator = Instrumentator(
        excluded_handlers=["/docs", "/redoc", "/metrics", "/openapi.json", "/favicon.ico", "none"]).instrument(app)

    if msd_overwritten is not None:
        cms_globals.model_service_dep = msd_overwritten

    cms_globals.props = Props(config.AUTH_USER_ENABLED == "true")

    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

    @app.on_event("startup")
    async def on_startup() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=50))
        RunVar("_default_thread_limiter").set(CapacityLimiter(50))  # type: ignore
        logger.debug("Default thread pool executor set to 50")
        instrumentator.expose(app, include_in_schema=False, should_gzip=False)
        logger.debug("Prometheus instrumentator metrics exposed")
        if config.AUTH_USER_ENABLED == "true":
            await make_sure_db_and_tables()
            logger.debug("Auth database and tables are ready")

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

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        TrackerClient.end_with_interruption()
        logger.debug("Tracker client terminated")

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        assert cms_globals.model_service_dep is not None, "Model service dependency not set"
        openapi_schema = get_openapi(
            title=f"{cms_globals.model_service_dep().model_name} APIs",
            version=cms_globals.model_service_dep().api_version,
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
        logger.debug("Custom OpenAPI schema generated")
        return app.openapi_schema

    return app


def _load_auth_router(app: FastAPI) -> FastAPI:
    from app.api.routers import authentication
    importlib.reload(authentication)
    app.include_router(authentication.router)
    return app


def _load_model_card(app: FastAPI) -> FastAPI:
    from app.api.routers import model_card
    importlib.reload(model_card)
    app.include_router(model_card.router)
    return app


def _load_invocation_router(app: FastAPI) -> FastAPI:
    from app.api.routers import invocation
    importlib.reload(invocation)
    app.include_router(invocation.router)
    return app


def _load_supervised_training_router(app: FastAPI) -> FastAPI:
    from app.api.routers import supervised_training
    importlib.reload(supervised_training)
    app.include_router(supervised_training.router)
    return app


def _load_evaluation_router(app: FastAPI) -> FastAPI:
    from app.api.routers import evaluation
    importlib.reload(evaluation)
    app.include_router(evaluation.router)
    return app


def _load_preview_router(app: FastAPI) -> FastAPI:
    from app.api.routers import preview
    importlib.reload(preview)
    app.include_router(preview.router)
    return app


def _load_unsupervised_training_router(app: FastAPI) -> FastAPI:
    from app.api.routers import unsupervised_training
    importlib.reload(unsupervised_training)
    app.include_router(unsupervised_training.router)
    return app


def _load_metacat_training_router(app: FastAPI) -> FastAPI:
    from app.api.routers import metacat_training
    importlib.reload(metacat_training)
    app.include_router(metacat_training.router)
    return app


def _load_training_operations(app: FastAPI) -> FastAPI:
    from app.api.routers import training_operations
    importlib.reload(training_operations)
    app.include_router(training_operations.router)
    return app

def _load_health_check_router(app: FastAPI) -> FastAPI:
    from app.api.routers import health_check
    importlib.reload(health_check)
    app.include_router(health_check.router)
    return app


def _load_stream_router(app: FastAPI) -> FastAPI:
    from app.api.routers import stream
    importlib.reload(stream)
    app.include_router(stream.router, prefix="/stream")
    return app
