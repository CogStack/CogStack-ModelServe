import os
import argparse
import logging.config
import json
import sys
import asyncio
import shutil
import warnings
import globals
from typing import Dict, Callable, Any, Optional
from urllib.parse import urlencode
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from hypercorn.config import Config
from hypercorn.asyncio import serve
from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from starlette.datastructures import QueryParams
from domain import Tags
from config import Settings
from monitoring.tracker import TrainingTracker
from monitoring.model_wrapper import ModelWrapper
from dependencies import ModelServiceDep

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.ini"), disable_existing_loggers=False)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


@lru_cache()
def get_settings():
    return Settings()


def get_model_server(msd_overwritten: Optional[ModelServiceDep] = None) -> FastAPI:
    if msd_overwritten is not None:
        globals.model_service_dep = msd_overwritten
    tags_metadata = [{"name": tag.name, "description": tag.value} for tag in Tags]
    app = FastAPI(openapi_tags=tags_metadata)

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
        TrainingTracker.end_with_interruption()

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=f"{model_service.model_name} APIs",
            version=model_service.api_version,
            description="by CogStack ModelServe, a model serving system for CogStack NLP solutions.",
            routes=app.routes
        )
        openapi_schema["info"]["x-logo"] = {
            "url": "https://avatars.githubusercontent.com/u/28688163?s=200&v=4"
        }
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    from routers import invocation
    app.include_router(invocation.router)

    if get_settings().ENABLE_TRAINING_APIS == "true":
        from routers import training
        app.include_router(training.router)

    app.openapi = custom_openapi  # type: ignore

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script serves various CogStack NLP models",
    )

    parser.add_argument(
        "-mt",
        "--model_type",
        help="The type of the model to serve",
        choices=["medcat", "de_id"],
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        help="The file path to the model package",
        type=str,
        default="",
    )

    parser.add_argument(
        "-mmu",
        "--mlflow_model_uri",
        help="The URI of the MLflow model to serve",
        type=str,
        default="",
    )

    parser.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="The hostname of the server",
    )

    parser.add_argument(
        "-p",
        "--port",
        default="8000",
        help="The port of the server",
    )

    parser.add_argument(
        "-d",
        "--doc",
        action="store_true",
        help="Export the OpenAPI doc",
    )

    args = parser.parse_args()
    settings = get_settings()

    model_service_dep = ModelServiceDep(args.model_type, settings)
    globals.model_service_dep = model_service_dep
    app = get_model_server()

    if args.doc:
        doc_name = ""
        if args.model_type == "medcat":
            doc_name = "medcat_model_apis.json"
        elif args.model_type == "de_id":
            doc_name = "de-dentification_model_apis.json"
        with open(doc_name, "w") as doc:
            json.dump(app.openapi(), doc, indent=4)
        print(f"OpenAPI doc exported to {doc_name}")
        sys.exit(0)
    else:
        dst_model_path = os.path.join(os.path.dirname(__file__), "model", "model.zip")
        if dst_model_path and os.path.exists(dst_model_path.replace(".zip", "")):
            shutil.rmtree(dst_model_path.replace(".zip", ""))
        if args.model_path:
            try:
                shutil.copy2(args.model_path, dst_model_path)
            except shutil.SameFileError:
                pass
            model_service = model_service_dep()
            model_service.init_model()
        elif args.mlflow_model_uri:
            model_service = ModelWrapper.get_model_service(settings.MLFLOW_TRACKING_URI, args.mlflow_model_uri)
            ModelWrapper.download_model_package(os.path.join(args.mlflow_model_uri, "artifacts"), dst_model_path)
            model_service_dep.model_service = model_service
            app = get_model_server()
        else:
            print("Error: Neither the model path or the mlflow model uri was passed in")
            sys.exit(1)

        config = Config()
        config.bind = [f"{args.host}:{args.port}"]
        config.access_log_format = "%(R)s %(s)s %(st)s %(D)s %({Header}o)s"
        config.accesslog = logger
        asyncio.run(serve(app, config))  # type: ignore
