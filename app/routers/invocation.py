from typing import Dict, List

from fastapi import APIRouter, Depends, Body, Request

import globals
from domain import TextwithAnnotations, ModelCard, Tags
from model_services.base import AbstractModelService
from utils import get_rate_limiter

router = APIRouter()
limiter = get_rate_limiter()


@router.get("/info", response_model=ModelCard, tags=[Tags.Metadata.name])
async def model_card(model_service: AbstractModelService = Depends(globals.model_service_dep)) -> ModelCard:
    return model_service.info()


@router.post("/process",
             response_model=TextwithAnnotations,
             response_model_exclude_none=True,
             tags=[Tags.Annotations.name])
@limiter.limit("60/minute")
def process_a_single_note(request: Request,
                          text: str = Body(..., media_type="text/plain"),
                          model_service: AbstractModelService = Depends(globals.model_service_dep)) -> Dict:
    annotations = model_service.annotate(text)
    return {"text": text, "annotations": annotations}


@router.post("/process_bulk",
             response_model=List[TextwithAnnotations],
             response_model_exclude_none=True,
             tags=[Tags.Annotations.name])
@limiter.limit("30/minute")
def process_a_list_of_notes(request: Request,
                            texts: List[str],
                            model_service: AbstractModelService = Depends(globals.model_service_dep)) -> List[Dict]:
    annotations_list = model_service.batch_annotate(texts)
    body = []
    for text, annotations in zip(texts, annotations_list):
        body.append({"text": text, "annotations": annotations})
    return body
