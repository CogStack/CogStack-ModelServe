import os
import argparse
import logging
import uvicorn
from functools import lru_cache
from fastapi import FastAPI
from typing import List, Dict
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
    async def info():
        return model_service.info()

    @app.post("/process", response_model=TextwithAnnotations, response_model_exclude_none=True)
    async def process(text: str):
        annotations = model_service.annotate(text)
        return {"text": text, "annotations": annotations}

    @app.post("/process_bulk", response_model=List[TextwithAnnotations], response_model_exclude_none=True)
    async def process_bulk(texts: List[str]):
        annotations_list = model_service.batch_annotate(texts)
        body = []
        for text, annotations in zip(texts, annotations_list):
            body.append({"text": text, "annotations": annotations})
        return body
    
    if hasattr(model_service, "train_supervised") and callable(model_service.train_supervised):
        @app.post("/trainsupervised")
        async def retrain(annotations: Dict):
            model_service.train_supervised(annotations)

    if hasattr(model_service, "train_unsupervised") and callable(model_service.train_unsupervised):
        @app.post("/trainunsupervised")
        async def retrain(texts: List[str]):
            model_service.train_unsupervised(texts)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script servers multiple cogstack models",
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
        raise f"Unknown model name: {args.model_name}"
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s %(levelname)s   %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s   %(message)s"
    logger.info(f'Start serving model "{args.model}" on {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=int(args.port), log_config=log_config)
