import argparse
import uvicorn
from fastapi import FastAPI
from domain import TextwithAnnotations
from typing import List
import config


def serve_model(modelrunner, host, port):
    app = FastAPI()

    @app.post("/process", response_model=TextwithAnnotations)
    def process(text: str):
        annotations = modelrunner.annotate(text)
        return {'text': text, 'annotations': annotations}

    @app.post("/process_bulk")
    def process_bulk(texts: List[str]):
        annotations = modelrunner.batchannotate(texts)
        print(annotations)

    @app.get("/info")
    def info():
        return {'model_description': 'medmen model', 'model_type': 'medcat'}
    
    if hasattr(modelrunner, "trainsupervised") and callable(modelrunner.trainsupervised):
        @app.post("/trainsupervised")
        def retrain(annotations: dict):
            modelrunner.trainsupervised(annotations)

    if hasattr(modelrunner, "trainunsupervised") and callable(modelrunner.trainunsupervised):
        @app.post("/trainunsupervised")
        def retrain(texts: List[str]):
            modelrunner.trainunsupervised(texts)
    
    uvicorn.run(app, host=host, port=port)


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
        from nlpmodel import NLPModel
        serve_model(NLPModel(config), args.host, int(args.port))
    else:
        raise f"Unknown model name: {args.model_name}"
