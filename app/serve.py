import os
import argparse
import logging.config
import uuid
import tempfile
import ijson
import json
import sys
import asyncio
import shutil
from enum import Enum
from typing import List, Dict, Callable, Any
from urllib.parse import urlencode
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from hypercorn.config import Config
from hypercorn.asyncio import serve
from fastapi import FastAPI, Request, Response, Body, File, UploadFile, Query
from fastapi.responses import HTMLResponse
from fastapi.openapi.utils import get_openapi
from starlette.datastructures import QueryParams
from starlette.status import HTTP_200_OK, HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE
from spacy import displacy
from domain import TextwithAnnotations, ModelCard, Doc
from model_services.base import AbstractModelService
from config import Settings
from utils import annotations_to_entities
from monitoring.tracker import TrainingTracker
from monitoring.model_wrapper import ModelWrapper

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.ini"), disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Tag(str, Enum):
    Metadata = "Get the model card."
    Annotations = "Retrieve recognised entities by running the model."
    Rendering = "Get embeddable annotation snippet in HTML."
    Training = "Trigger model training on input annotations."


@lru_cache()
def get_settings():
    return Settings()


def get_model_server(model_service: AbstractModelService) -> FastAPI:
    tags_metadata = [{"name": tag.name, "description": tag.value} for tag in Tag]
    app = FastAPI(penapi_tags=tags_metadata)

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

    @app.get("/info", response_model=ModelCard, tags=[Tag.Metadata.name])
    async def model_card() -> ModelCard:
        return model_service.info()

    @app.post("/process",
              response_model=TextwithAnnotations,
              response_model_exclude_none=True,
              tags=[Tag.Annotations.name])
    async def process_a_single_note(text: str = Body(..., media_type="text/plain")) -> Dict:
        annotations = model_service.annotate(text)
        return {"text": text, "annotations": annotations}

    @app.post("/process_bulk",
              response_model=List[TextwithAnnotations],
              response_model_exclude_none=True,
              tags=[Tag.Annotations.name])
    async def process_a_list_of_notes(texts: List[str]) -> List[Dict]:
        annotations_list = model_service.batch_annotate(texts)
        body = []
        for text, annotations in zip(texts, annotations_list):
            body.append({"text": text, "annotations": annotations})
        return body

    @app.post("/preview", tags=[Tag.Rendering.name], response_class=HTMLResponse)
    async def preview_processing_result(text: str = Body(..., media_type="text/plain")) -> HTMLResponse:
        annotations = model_service.annotate(text)
        entities = annotations_to_entities(annotations)
        ent_input = Doc(text=text, ents=entities)
        data = displacy.render(ent_input.dict(), style="ent", manual=True)
        response = HTMLResponse(content=data, status_code=HTTP_200_OK)
        response.headers["Content-Disposition"] = f'attachment ; filename="{str(uuid.uuid4())}.html"'
        return response

    if hasattr(model_service, "train_supervised") and callable(model_service.train_supervised):
        @app.post("/train_supervised", status_code=HTTP_202_ACCEPTED, tags=[Tag.Training.name])
        async def supervised_training(training_data: UploadFile,
                                      response: Response,
                                      epochs: int = 1,
                                      log_frequency: int = Query(default=1, description="log after every number of finished epochs")) -> Dict:
            data_file = tempfile.NamedTemporaryFile()
            for line in training_data.file:
                data_file.write(line)
            data_file.flush()
            training_id = str(uuid.uuid4())
            training_accepted = model_service.train_supervised(data_file, epochs, log_frequency, training_id, training_data.filename)
            return _get_training_response(training_accepted, response, training_id)

    if hasattr(model_service, "train_unsupervised") and callable(model_service.train_unsupervised):
        @app.post("/train_unsupervised", status_code=HTTP_202_ACCEPTED, tags=[Tag.Training.name])
        async def unsupervised_training(response: Response,
                                        training_data: UploadFile = File(...),
                                        log_frequency: int = Query(default=1000, description="log after every number of processed documents")) -> Dict:
            texts = ijson.items(training_data.file, "item")
            training_id = str(uuid.uuid4())
            training_accepted = model_service.train_unsupervised(texts, 1, log_frequency, training_id, training_data.filename)
            return _get_training_response(training_accepted, response, training_id)

    def _get_training_response(training_accepted: bool, response: Response, training_id: str) -> Dict:
        if training_accepted:
            return {"message": "Your training started successfully.", "training_id": training_id}
        else:
            response.status_code = HTTP_503_SERVICE_UNAVAILABLE
            return {"message": "Another training is in progress. Please retry your training later."}

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

    app.openapi = custom_openapi  # type: ignore

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script serves various CogStack NLP models",
    )

    parser.add_argument(
        "-m",
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

    if args.model_type == "medcat":
        from model_services.medcat_model import MedCATModel
        model_service = MedCATModel(settings)
        app = get_model_server(model_service)
    elif args.model_type == "de_id":
        from model_services.deid_model import DeIdModel
        model_service = DeIdModel(settings)
        app = get_model_server(model_service)
    else:
        print(f"Error: unknown model type: {args.model_type}")
        sys.exit(1)

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
        if args.model_path:
            try:
                dst_model_path = os.path.join(os.path.dirname(__file__), "model", "model.zip")
                shutil.copy2(args.model_path, dst_model_path)
            except shutil.SameFileError:
                pass
            if dst_model_path and os.path.exists(dst_model_path.replace(".zip", "")):
                shutil.rmtree(dst_model_path.replace(".zip", ""))
            model_service.init_model()
        elif args.mlflow_model_uri:
            model_service = ModelWrapper.get_model_service(settings.MLFLOW_TRACKING_URI, args.mlflow_model_uri)
            app = get_model_server(model_service)
            # TODO: replace /app/model/model.zip with the model artifact
        else:
            print("Error: Neither the model path or the mlflow model uri was passed in")
            sys.exit(1)

        config = Config()
        config.bind = [f"{args.host}:{args.port}"]
        config.access_log_format = "%(R)s %(s)s %(st)s %(D)s %({Header}o)s"
        config.accesslog = logger
        asyncio.run(serve(app, config))  # type: ignore
