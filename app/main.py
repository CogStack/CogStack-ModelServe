from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from nlpmodel import modelrunner

app = FastAPI()
modelrunner = modelrunner()


class Annotation(BaseModel):
    start: int
    end: int
    label_name: str
    label_id: str
    meta_anns: Optional[dict] = None
    

class TextwithAnnotations(BaseModel):
    text: str
    annotations: List[Annotation] 

@app.post("/process", response_model=TextwithAnnotations)
def process(text: str):
    annotations = modelrunner.annotate(text)
    return {'text': text, 'annotations': annotations}

@app.post("/process_bulk")
def process_bulk(texts: List[str]):
    annotations = modelrunner.batchannotate(texts)
    print(annotations)

@app.post("/trainsupervised")
def retrain(annotations: dict):
    modelrunner.trainsupervised(annotations)

@app.post("/trainunsupervised")
def retrain(texts: List[str]):
    modelrunner.trainunsupervised(texts)

@app.get("/info")
def info():

    return {'model_description': 'medmen model', 'model_type': 'medcat'}


