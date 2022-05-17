import argparse
import uvicorn
from fastapi import FastAPI
from typing import List, Dict
from domain import TextwithAnnotations, ModelCard
from model_services.base import AbstractModelService
import config


def get_model_server(modelrunner: AbstractModelService) -> FastAPI:
    app = FastAPI()

    @app.get("/info", response_model=ModelCard)
    async def info():
        return modelrunner.info()

    @app.post("/process", response_model=TextwithAnnotations, response_model_exclude_none=True)
    async def process(text: str):
        annotations = modelrunner.annotate(text)
        return {'text': text, 'annotations': annotations}

    @app.post("/process_bulk", response_model=List[TextwithAnnotations], response_model_exclude_none=True)
    async def process_bulk(texts: List[str]):
        annotations_list = modelrunner.batch_annotate(texts)
        body = []
        for text, annotations in zip(texts, annotations_list):
            body.append({'text': text, 'annotations': annotations})
        return body
    
    if hasattr(modelrunner, "train_supervised") and callable(modelrunner.train_supervised):
        @app.post("/trainsupervised")
        async def retrain(annotations: Dict):
            modelrunner.train_supervised(annotations)

    if hasattr(modelrunner, "train_unsupervised") and callable(modelrunner.train_unsupervised):
        @app.post("/trainunsupervised")
        async def retrain(texts: List[str]):
            modelrunner.train_unsupervised(texts)

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
        app = get_model_server(NlpModel(config))
    elif args.model == "de_id":
        from model_services.deid_model import DeIdModel
        app = get_model_server(DeIdModel(config))
    else:
        raise f"Unknown model name: {args.model_name}"
    uvicorn.run(app, host=args.host, port=int(args.port))
