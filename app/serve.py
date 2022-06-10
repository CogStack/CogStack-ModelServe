import os
import argparse
import logging.config
import uvicorn
import uuid
import tempfile
import ijson
import json
import sys
import mlflow
from enum import Enum
from typing import List, Dict, Callable, Any
from urllib.parse import urlencode
from functools import lru_cache
from fastapi import FastAPI, Request, Response, Body, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.openapi.utils import get_openapi
from starlette.datastructures import QueryParams
from starlette.status import HTTP_201_CREATED, HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE
from spacy import displacy
from mlflow.entities import RunStatus
from domain import TextwithAnnotations, ModelCard, Doc
from model_services.base import AbstractModelService
from config import Settings
from utils import annotations_to_entities

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

    @app.post("/preview", tags=[Tag.Rendering.name])
    async def preview_processing_result(text: str = Body(..., media_type="text/plain")) -> HTMLResponse:
        annotations = model_service.annotate(text)
        entities = annotations_to_entities(annotations)
        ent_input = Doc(text=text, ents=entities)
        data = displacy.render(ent_input.dict(), style="ent", manual=True)
        response = HTMLResponse(content=data, status_code=HTTP_201_CREATED)
        response.headers["Content-Disposition"] = f'attachment ; filename="processed_{str(uuid.uuid4())}.html"'
        return response

    if hasattr(model_service, "train_supervised") and callable(model_service.train_supervised):
        @app.post("/train_supervised", status_code=HTTP_202_ACCEPTED, tags=[Tag.Training.name])
        async def supervised_training(file: UploadFile,
                                      response: Response,
                                      epochs: int = 1,
                                      redeploy: bool = False,
                                      skip_save_model: bool = True) -> Dict:
            data_file = tempfile.NamedTemporaryFile()
            for line in file.file:
                data_file.write(line)
            data_file.flush()
            correlation_id = str(uuid.uuid4())
            training_accepted = model_service.train_supervised(data_file, epochs, redeploy, skip_save_model, correlation_id)
            return _get_training_response(training_accepted, response, correlation_id)

    if hasattr(model_service, "train_unsupervised") and callable(model_service.train_unsupervised):
        @app.post("/train_unsupervised", status_code=HTTP_202_ACCEPTED, tags=[Tag.Training.name])
        async def unsupervised_training(response: Response,
                                        file: UploadFile = File(...),
                                        epochs: int = 1,
                                        redeploy: bool = False,
                                        skip_save_model: bool = True) -> Dict:
            texts = ijson.items(file.file, "item")
            correlation_id = str(uuid.uuid4())
            training_accepted = model_service.train_unsupervised(texts, epochs, redeploy, skip_save_model, correlation_id)
            return _get_training_response(training_accepted, response, correlation_id)

    def _get_training_response(training_accepted: bool, response: Response, correlation_id: str) -> Dict:
        if training_accepted:
            return {"message": "Your training started successfully.", "correlation_id": correlation_id}
        else:
            response.status_code = HTTP_503_SERVICE_UNAVAILABLE
            return {"message": "Another training is in progress. Please retry your training later."}

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
        mlflow.end_run(RunStatus.to_string(RunStatus.KILLED))

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=f"{model_service.info().model_description.title()} APIs",
            version=model_service.info().api_version,
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
        "--model",
        help="The name of the model to serve",
        choices=["medcat_1_2", "de_id"],
        required=True
    )

    parser.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="The hostname of the server"
    )

    parser.add_argument(
        "-p",
        "--port",
        default="8000",
        help="The port of the server"
    )

    parser.add_argument(
        "-d",
        "--doc",
        action="store_true",
        help="Export the OpenAPI doc"
    )

    args = parser.parse_args()
    if args.model == "medcat_1_2":
        from model_services.medcat_model import MedCATModel
        app = get_model_server(MedCATModel(get_settings()))
    elif args.model == "de_id":
        from model_services.deid_model import DeIdModel
        app = get_model_server(DeIdModel(get_settings()))

    if args.doc:
        doc_name = ""
        if args.model == "medcat_1_2":
            if get_settings().CODE_TYPE == "snomed":
                doc_name = "snomed_model_apis.json"
            elif get_settings().CODE_TYPE == "icd10":
                doc_name = "icd10_model_apis.json"
            else:
                raise ValueError(f"Unknown code type: {get_settings().CODE_TYPE}")
        elif args.model == "de_id":
            doc_name = "de-dentification_model_apis.json"
        with open(doc_name, "w") as doc:
            json.dump(app.openapi(), doc, indent=4)
        print(f"OpenAPI doc exported to {doc_name}")
        sys.exit(0)

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelname)s   %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s   %(message)s"
    logger.info(f'Start serving model "{args.model}" on {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=int(args.port), log_config=log_config)
