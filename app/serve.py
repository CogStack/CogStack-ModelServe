import os
import argparse
import logging.config
import uvicorn
from urllib.parse import urlencode
from functools import lru_cache
from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from starlette.datastructures import QueryParams
from typing import List, Dict, Callable, Any
from domain import TextwithAnnotations, ModelCard
from model_services.base import AbstractModelService
from config import Settings

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.ini"), disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@lru_cache()
def get_settings():
    return Settings()


def get_model_server(model_service: AbstractModelService) -> FastAPI:
    app = FastAPI()

    @app.get("/info", response_model=ModelCard)
    async def info() -> ModelCard:
        return model_service.info()

    @app.post("/process", response_model=TextwithAnnotations, response_model_exclude_none=True)
    async def process(text: str) -> Dict:
        annotations = model_service.annotate(text)
        return {"text": text, "annotations": annotations}

    @app.post("/process_bulk", response_model=List[TextwithAnnotations], response_model_exclude_none=True)
    async def process_bulk(texts: List[str]) -> List[Dict]:
        annotations_list = model_service.batch_annotate(texts)
        body = []
        for text, annotations in zip(texts, annotations_list):
            body.append({"text": text, "annotations": annotations})
        return body
    
    if hasattr(model_service, "train_supervised") and callable(model_service.train_supervised):
        @app.post("/trainsupervised")
        async def retrain(annotations: Dict) -> None:
            model_service.train_supervised(annotations)

    if hasattr(model_service, "train_unsupervised") and callable(model_service.train_unsupervised):
        @app.post("/trainunsupervised")
        async def retrain_unsupervised(texts: List[str]) -> None:
            model_service.train_unsupervised(texts)

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

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=f"{model_service.info().model_description.title()} APIs",
            version="0.0.1",
            description="by CogStack Model Farm, a model serving system for CogStack NLP solutions.",
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

    args = parser.parse_args()
    if args.model == "medcat_1_2":
        from model_services.nlp_model import NlpModel
        app = get_model_server(NlpModel(get_settings()))
    elif args.model == "de_id":
        from model_services.deid_model import DeIdModel
        app = get_model_server(DeIdModel(get_settings()))
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelname)s   %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s   %(message)s"
    logger.info(f'Start serving model "{args.model}" on {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=int(args.port), log_config=log_config)
