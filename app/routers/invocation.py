from typing import Dict, List

from fastapi import APIRouter, Depends, Body, Request

import globals
from domain import TextwithAnnotations, ModelCard, Tags
from model_services.base import AbstractModelService
from utils import get_settings, get_rate_limiter
from management.prometheus_metrics import cms_doc_annotations

PATH_INFO = "/info"
PATH_PROCESS = "/process"
PATH_PROCESS_BULK = "/process_bulk"

router = APIRouter()
limiter = get_rate_limiter()


@router.get(PATH_INFO, response_model=ModelCard, tags=[Tags.Metadata.name])
async def model_card(model_service: AbstractModelService = Depends(globals.model_service_dep)) -> ModelCard:
    return model_service.info()


@router.post(PATH_PROCESS,
             response_model=TextwithAnnotations,
             response_model_exclude_none=True,
             tags=[Tags.Annotations.name])
@limiter.limit(get_settings().PROCESS_RATE_LIMIT)
def process_a_single_note(request: Request,
                          text: str = Body(..., media_type="text/plain"),
                          model_service: AbstractModelService = Depends(globals.model_service_dep)) -> Dict:
    annotations = model_service.annotate(text)
    cms_doc_annotations.labels(handler=PATH_PROCESS).observe(len(annotations))

    return {"text": text, "annotations": annotations}


@router.post(PATH_PROCESS_BULK,
             response_model=List[TextwithAnnotations],
             response_model_exclude_none=True,
             tags=[Tags.Annotations.name])
@limiter.limit(get_settings().PROCESS_BULK_RATE_LIMIT)
def process_a_list_of_notes(request: Request,
                            texts: List[str],
                            model_service: AbstractModelService = Depends(globals.model_service_dep)) -> List[Dict]:
    annotations_list = model_service.batch_annotate(texts)
    body = []
    annotation_sum = 0
    for text, annotations in zip(texts, annotations_list):
        body.append({"text": text, "annotations": annotations})
        annotation_sum += len(annotations)
    cms_doc_annotations.labels(handler=PATH_PROCESS_BULK).observe(annotation_sum)

    return body
