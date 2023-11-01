import statistics
import tempfile
from io import BytesIO

import json
import ijson
import uuid
import hashlib
from typing import Dict, List, Union
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, Body, UploadFile, File, Request, Query
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse

import globals
from domain import TextWithAnnotations, ModelCard, Tags
from model_services.base import AbstractModelService
from utils import get_settings, get_rate_limiter
from management.prometheus_metrics import (
    cms_doc_annotations,
    cms_avg_anno_acc_per_doc,
    cms_avg_meta_anno_conf_per_doc,
    cms_bulk_processed_docs,
)
from auth.users import props
from processors.data_batcher import mini_batch

PATH_INFO = "/info"
PATH_PROCESS = "/process"
PATH_PROCESS_BULK = "/process_bulk"
PATH_PROCESS_BULK_FILE = "/process_bulk_file"
PATH_REDACT = "/redact"

router = APIRouter()
limiter = get_rate_limiter()


@router.get(PATH_INFO,
            response_model=ModelCard,
            tags=[Tags.Metadata.name],
            dependencies=[Depends(props.current_active_user)],
            description="Get information about the model being served")
async def get_model_card(request: Request,
                         model_service: AbstractModelService = Depends(globals.model_service_dep)) -> ModelCard:
    return model_service.info()


@router.post(PATH_PROCESS,
             response_model=TextWithAnnotations,
             response_model_exclude_none=True,
             response_class=JSONResponse,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(props.current_active_user)],
             description="Extract the NER entities from a single piece of plain text")
@limiter.limit(get_settings().PROCESS_RATE_LIMIT)
def get_entities_from_text(request: Request,
                           text: Annotated[str, Body(description="The plain text to be sent to the model for NER", media_type="text/plain")],
                           model_service: AbstractModelService = Depends(globals.model_service_dep)) -> TextWithAnnotations:
    annotations = model_service.annotate(text)
    _send_annotation_num_metric(len(annotations), PATH_PROCESS)

    _send_accuracy_metric(annotations, PATH_PROCESS)
    _send_confidence_metric(annotations, PATH_PROCESS)

    return TextWithAnnotations(text=text, annotations=annotations)


@router.post(PATH_PROCESS_BULK,
             response_model=List[TextWithAnnotations],
             response_model_exclude_none=True,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(props.current_active_user)],
             description="Extract the NER entities from multiple plain texts")
@limiter.limit(get_settings().PROCESS_BULK_RATE_LIMIT)
def get_entities_from_multiple_texts(request: Request,
                                     texts: Annotated[List[str], Body(description="A list of plain texts to be sent to the model for NER, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
                                     model_service: AbstractModelService = Depends(globals.model_service_dep)) -> List[TextWithAnnotations]:
    annotations_list = model_service.batch_annotate(texts)
    body = []
    annotation_sum = 0
    for text, annotations in zip(texts, annotations_list):
        body.append({"text": text, "annotations": annotations})
        annotation_sum += len(annotations)
        _send_accuracy_metric(annotations, PATH_PROCESS_BULK)
        _send_confidence_metric(annotations, PATH_PROCESS_BULK)

    _send_bulk_processed_docs_metric(body, PATH_PROCESS_BULK)
    _send_annotation_num_metric(annotation_sum, PATH_PROCESS_BULK)

    return body


@router.post(PATH_PROCESS_BULK_FILE,
             tags=[Tags.Annotations.name],
             response_class=StreamingResponse,
             dependencies=[Depends(props.current_active_user)],
             description="Upload a file containing a list of plain text and extract the NER entities in JSON")
def extract_entities_from_multi_text_file(request: Request,
                                          multi_text_file: Annotated[UploadFile, File(description="A file containing a list of plain texts, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
                                          model_service: AbstractModelService = Depends(globals.model_service_dep)) -> StreamingResponse:
    with tempfile.NamedTemporaryFile() as data_file:
        for line in multi_text_file.file:
            data_file.write(line)
        data_file.flush()

        data_file.seek(0)
        texts = ijson.items(data_file, "item")
        annotations_list = []
        for batch in mini_batch(texts, batch_size=5):
            annotations_list += model_service.batch_annotate(batch)

        body = []
        annotation_sum = 0
        data_file.seek(0)
        texts = ijson.items(data_file, "item")
        for text, annotations in zip(texts, annotations_list):
            body.append({"text": text, "annotations": annotations})
            annotation_sum += len(annotations)
            _send_accuracy_metric(annotations, PATH_PROCESS_BULK)
            _send_confidence_metric(annotations, PATH_PROCESS_BULK)

        _send_bulk_processed_docs_metric(body, PATH_PROCESS_BULK)
        _send_annotation_num_metric(annotation_sum, PATH_PROCESS_BULK)

        json_file = BytesIO(json.dumps(body).encode())
        response = StreamingResponse(json_file, media_type="application/json")
        response.headers["Content-Disposition"] = f'attachment ; filename="concatenated_{str(uuid.uuid4())}.json"'
        return response


@router.post(PATH_REDACT,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(props.current_active_user)],
             description="Extract and redact NER entities from a single piece of plain text")
@limiter.limit(get_settings().PROCESS_RATE_LIMIT)
def get_redacted_text(request: Request,
                      text: Annotated[str, Body(description="The plain text to be sent to the model for NER and redaction", media_type="text/plain")],
                      mask: Annotated[Union[str, None], Query(description="The custom symbols used for masking detected spans")] = None,
                      hash: Annotated[Union[bool, None], Query(description="Whether or not to hash detected spans")] = False,
                      model_service: AbstractModelService = Depends(globals.model_service_dep)) -> PlainTextResponse:
    annotations = model_service.annotate(text)
    _send_annotation_num_metric(len(annotations), PATH_PROCESS)

    _send_accuracy_metric(annotations, PATH_PROCESS)
    _send_confidence_metric(annotations, PATH_PROCESS)

    redacted_text = ""
    start_index = 0
    for annotation in annotations:
        if hash:
            label = hashlib.sha256(text[annotation["start"]:annotation["end"]].encode()).hexdigest()
        elif mask is None or len(mask) == 0:
            label = f"[{annotation['label_name']}]"
        else:
            label = mask
        redacted_text += text[start_index:annotation["start"]] + label
        start_index = annotation["end"]
    redacted_text += text[start_index:]

    return PlainTextResponse(redacted_text)


def _send_annotation_num_metric(annotation_num: int, handler: str) -> None:
    cms_doc_annotations.labels(handler=handler).observe(annotation_num)


def _send_accuracy_metric(annotations: List[Dict], handler: str) -> None:
    if annotations and annotations[0].get("accuracy", None) is not None:
        avg_acc = statistics.mean([annotation["accuracy"] for annotation in annotations])
        cms_avg_anno_acc_per_doc.labels(handler=handler).set(avg_acc)


def _send_confidence_metric(annotations: List[Dict], handler: str) -> None:
    if annotations and annotations[0].get("meta_anns", None):
        avg_conf = statistics.mean([meta_value["confidence"] for annotation in annotations for _, meta_value in annotation["meta_anns"].items()])
        cms_avg_meta_anno_conf_per_doc.labels(handler=handler).set(avg_conf)


def _send_bulk_processed_docs_metric(processed_docs: List[Dict], handler: str) -> None:
    cms_bulk_processed_docs.labels(handler=handler).observe(len(processed_docs))
