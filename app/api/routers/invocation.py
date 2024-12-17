import statistics
import tempfile
import itertools
import json
import ijson
import uuid
import hashlib
import logging
import pandas as pd
import api.globals as cms_globals

from typing import Dict, List, Union, Iterator, Any
from collections import defaultdict
from io import BytesIO
from starlette.status import HTTP_400_BAD_REQUEST
from typing_extensions import Annotated
from fastapi import APIRouter, Depends, Body, UploadFile, File, Request, Query, Response
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import ValidationError
from domain import TextWithAnnotations, TextWithPublicKey, TextStreamItem, ModelCard, Tags
from model_services.base import AbstractModelService
from utils import get_settings
from api.utils import get_rate_limiter, encrypt
from management.prometheus_metrics import (
    cms_doc_annotations,
    cms_avg_anno_acc_per_doc,
    cms_avg_anno_acc_per_concept,
    cms_avg_meta_anno_conf_per_doc,
    cms_bulk_processed_docs,
)
from processors.data_batcher import mini_batch

PATH_INFO = "/info"
PATH_PROCESS = "/process"
PATH_PROCESS_JSON_LINES = "/process_jsonl"
PATH_PROCESS_BULK = "/process_bulk"
PATH_PROCESS_BULK_FILE = "/process_bulk_file"
PATH_REDACT = "/redact"
PATH_REDACT_WITH_ENCRYPTION = "/redact_with_encryption"

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)

logger = logging.getLogger("cms")


@router.get(PATH_INFO,
            response_model=ModelCard,
            tags=[Tags.Metadata.name],
            dependencies=[Depends(cms_globals.props.current_active_user)],
            description="Get information about the model being served")
async def get_model_card(request: Request,
                         model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> ModelCard:
    return model_service.info()


@router.post(PATH_PROCESS,
             response_model=TextWithAnnotations,
             response_model_exclude_none=True,
             response_class=JSONResponse,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Extract the NER entities from a single piece of plain text")
@limiter.limit(config.PROCESS_RATE_LIMIT)
def get_entities_from_text(request: Request,
                           text: Annotated[str, Body(description="The plain text to be sent to the model for NER", media_type="text/plain")],
                           model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> TextWithAnnotations:
    annotations = model_service.annotate(text)
    _send_annotation_num_metric(len(annotations), PATH_PROCESS)

    _send_accuracy_metric(annotations, PATH_PROCESS)
    _send_meta_confidence_metric(annotations, PATH_PROCESS)

    logger.debug(annotations)
    return TextWithAnnotations(text=text, annotations=annotations)


@router.post(PATH_PROCESS_JSON_LINES,
             response_class=StreamingResponse,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Extract the NER entities from texts in the JSON Lines format")
@limiter.limit(config.PROCESS_RATE_LIMIT)
def get_entities_from_jsonlines_text(request: Request,
                                     json_lines: Annotated[str, Body(description="The texts in the jsonlines format and each line contains {\"text\": \"<TEXT>\"[, \"name\": \"<NAME>\"]}", media_type="application/x-ndjson")]) -> Response:
    model_manager = cms_globals.model_manager_dep()
    stream: Iterator[Dict[str, Any]] = itertools.chain()

    try:
        for chunked_input in _chunk_request_body(json_lines):
            predicted_stream = model_manager.predict_stream(context=None, model_input=chunked_input)
            stream = itertools.chain(stream, predicted_stream)

        return StreamingResponse(_get_jsonlines_stream(stream), media_type="application/x-ndjson; charset=utf-8")
    except json.JSONDecodeError:
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": "Invalid JSON Lines."})
    except ValidationError:
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"message": f"Invalid JSON properties found. The schema should be {TextStreamItem.schema_json()}"})


@router.post(PATH_PROCESS_BULK,
             response_model=List[TextWithAnnotations],
             response_model_exclude_none=True,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Extract the NER entities from multiple plain texts")
@limiter.limit(config.PROCESS_BULK_RATE_LIMIT)
def get_entities_from_multiple_texts(request: Request,
                                     texts: Annotated[List[str], Body(description="A list of plain texts to be sent to the model for NER, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
                                     model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> List[TextWithAnnotations]:
    annotations_list = model_service.batch_annotate(texts)
    body: List[TextWithAnnotations] = []
    annotation_sum = 0
    for text, annotations in zip(texts, annotations_list):
        body.append(TextWithAnnotations(text=text, annotations=annotations))
        annotation_sum += len(annotations)
        _send_accuracy_metric(annotations, PATH_PROCESS_BULK)
        _send_meta_confidence_metric(annotations, PATH_PROCESS_BULK)

    _send_bulk_processed_docs_metric(body, PATH_PROCESS_BULK)
    _send_annotation_num_metric(annotation_sum, PATH_PROCESS_BULK)

    logger.debug(body)
    return body


@router.post(PATH_PROCESS_BULK_FILE,
             tags=[Tags.Annotations.name],
             response_class=StreamingResponse,
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Upload a file containing a list of plain text and extract the NER entities in JSON")
def extract_entities_from_multi_text_file(request: Request,
                                          multi_text_file: Annotated[UploadFile, File(description="A file containing a list of plain texts, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
                                          tracking_id: Annotated[Union[str, None], Query(description="The tracking ID of the bulk processing task")] = None,
                                          model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> StreamingResponse:
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
            _send_meta_confidence_metric(annotations, PATH_PROCESS_BULK)

        _send_bulk_processed_docs_metric(body, PATH_PROCESS_BULK)
        _send_annotation_num_metric(annotation_sum, PATH_PROCESS_BULK)

        output = json.dumps(body)
        logger.debug(output)
        json_file = BytesIO(output.encode())
        tracking_id = tracking_id or str(uuid.uuid4())
        response = StreamingResponse(json_file, media_type="application/json")
        response.headers["Content-Disposition"] = f'attachment ; filename="concatenated_{tracking_id}.json"'
        return response


@router.post(PATH_REDACT,
             tags=[Tags.Redaction.name],
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Extract and redact NER entities from a single piece of plain text")
@limiter.limit(config.PROCESS_RATE_LIMIT)
def get_redacted_text(request: Request,
                      text: Annotated[str, Body(description="The plain text to be sent to the model for NER and redaction", media_type="text/plain")],
                      concepts_to_keep: Annotated[List[str], Query(description="List of concepts (Label IDs) that should not be removedd during the redaction process. List should be in the format ['label1','label2'...]")] = [],
                      warn_on_no_redaction: Annotated[Union[bool, None], Query(description="Return warning when no entities were detected for redaction to prevent potential info leaking")] = False,
                      mask: Annotated[Union[str, None], Query(description="The custom symbols used for masking detected spans")] = None,
                      hash: Annotated[Union[bool, None], Query(description="Whether or not to hash detected spans")] = False,
                      model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> PlainTextResponse:
    annotations = model_service.annotate(text)
    _send_annotation_num_metric(len(annotations), PATH_REDACT)

    _send_accuracy_metric(annotations, PATH_REDACT)
    _send_meta_confidence_metric(annotations, PATH_REDACT)

    redacted_text = ""
    start_index = 0
    if not annotations and warn_on_no_redaction:
        return PlainTextResponse(content="WARNING: No entities were detected for redaction.", status_code=200)
    else:
        for annotation in annotations:

            if annotation["label_id"] in concepts_to_keep:
                continue

            if hash:
                label = hashlib.sha256(text[annotation["start"]:annotation["end"]].encode()).hexdigest()
            elif mask is None or len(mask) == 0:
                label = f"[{annotation['label_name']}]"
            else:
                label = mask
            redacted_text += text[start_index:annotation["start"]] + label
            start_index = annotation["end"]
        redacted_text += text[start_index:]
        logger.debug(redacted_text)
        return PlainTextResponse(content=redacted_text, status_code=200)


@router.post(PATH_REDACT_WITH_ENCRYPTION,
             tags=[Tags.Redaction.name],
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Redact and encrypt NER entities from a single piece of plain text")
@limiter.limit(config.PROCESS_RATE_LIMIT)
def get_redacted_text_with_encryption(request: Request,
                                      text_with_public_key: Annotated[TextWithPublicKey, Body()],
                                      warn_on_no_redaction: Annotated[Union[bool, None], Query(description="Return warning when no entities were detected for redaction to prevent potential info leaking")] = False,
                                      model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> JSONResponse:
    annotations = model_service.annotate(text_with_public_key.text)
    _send_annotation_num_metric(len(annotations), PATH_REDACT_WITH_ENCRYPTION)

    _send_accuracy_metric(annotations, PATH_REDACT_WITH_ENCRYPTION)
    _send_meta_confidence_metric(annotations, PATH_REDACT_WITH_ENCRYPTION)

    redacted_text = ""
    start_index = 0
    encryptions = []
    if not annotations and warn_on_no_redaction:
        return JSONResponse(content={"message": "WARNING: No entities were detected for redaction."})
    else:
        for idx, annotation in enumerate(annotations):
            label = f"[REDACTED_{idx}]"
            encrypted = encrypt(text_with_public_key.text[annotation["start"]:annotation["end"]], text_with_public_key.public_key_pem)
            redacted_text += text_with_public_key.text[start_index:annotation["start"]] + label
            encryptions.append({"label": label, "encryption": encrypted})
            start_index = annotation["end"]
        redacted_text += text_with_public_key.text[start_index:]

        content = {"redacted_text": redacted_text, "encryptions": encryptions}
        logger.debug(content)
        return JSONResponse(content=content)


def _send_annotation_num_metric(annotation_num: int, handler: str) -> None:
    cms_doc_annotations.labels(handler=handler).observe(annotation_num)


def _send_accuracy_metric(annotations: List[Dict], handler: str) -> None:
    if annotations and annotations[0].get("accuracy", None) is not None:
        doc_avg_acc = statistics.mean([annotation["accuracy"] for annotation in annotations])
        cms_avg_anno_acc_per_doc.labels(handler=handler).set(doc_avg_acc)

        if config.LOG_PER_CONCEPT_ACCURACIES == "true":
            accumulated_concept_accuracy: Dict[str, float] = defaultdict(float)
            concept_count: Dict[str, int] = defaultdict(int)
            for annotation in annotations:
                accumulated_concept_accuracy[annotation["label_id"]] += annotation["accuracy"]
                concept_count[annotation["label_id"]] += 1
            for concept, accumulated_accuracy in accumulated_concept_accuracy.items():
                concept_avg_acc = accumulated_accuracy / concept_count[concept]
                cms_avg_anno_acc_per_concept.labels(handler=handler, concept=concept).set(concept_avg_acc)


def _send_meta_confidence_metric(annotations: List[Dict], handler: str) -> None:
    if annotations and annotations[0].get("meta_anns", None):
        avg_conf = statistics.mean([meta_value["confidence"] for annotation in annotations for _, meta_value in annotation["meta_anns"].items()])
        cms_avg_meta_anno_conf_per_doc.labels(handler=handler).set(avg_conf)


def _send_bulk_processed_docs_metric(processed_docs: List[Dict], handler: str) -> None:
    cms_bulk_processed_docs.labels(handler=handler).observe(len(processed_docs))


def _chunk_request_body(json_lines: str, chunk_size: int = 5) -> Iterator[pd.DataFrame]:
    chunk = []
    doc_idx = 0
    for line in json_lines.splitlines():
        json_line_obj = json.loads(line)
        TextStreamItem(**json_line_obj)
        if "name" not in json_line_obj:
            json_line_obj["name"] = str(doc_idx)
        doc_idx += 1
        chunk.append(json_line_obj)

        if len(chunk) == chunk_size:
            df = pd.DataFrame(chunk)
            yield df
            chunk.clear()
    if chunk:
        df = pd.DataFrame(chunk)
        yield df
        chunk.clear()


def _get_jsonlines_stream(output_stream: Iterator[Dict[str, Any]]) -> Iterator[str]:
    current_doc_name = ""
    annotation_num = 0
    for item in output_stream:
        if current_doc_name != "" and current_doc_name != item["doc_name"]:
            cms_doc_annotations.labels(handler=PATH_PROCESS_JSON_LINES).observe(annotation_num)
        current_doc_name = item["doc_name"]
        annotation_num += 1
        line = json.dumps(item) + "\n"
        logger.debug(line)
        yield json.dumps(item) + "\n"
