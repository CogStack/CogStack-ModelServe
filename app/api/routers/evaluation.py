import io
import json
import uuid
import tempfile
import pandas as pd

from typing import List, Dict, Any, Union, cast
from typing_extensions import Annotated
from fastapi import APIRouter, Query, Depends, UploadFile, Request, File
from fastapi.responses import StreamingResponse, JSONResponse

import app.api.globals as cms_globals
from app.api.dependencies import validate_tracking_id
from app.domain import Tags, Scope
from app.model_services.base import AbstractModelService
from app.processors.metrics_collector import (
    sanity_check_model_with_trainer_export,
    get_iaa_scores_per_concept,
    get_iaa_scores_per_doc,
    get_iaa_scores_per_span,
    concat_trainer_exports,
    get_stats_from_trainer_export,
)
from app.exception import AnnotationException
from app.utils import filter_by_concept_ids

router = APIRouter()

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.post(
    "/sanity-check",
    tags=[Tags.Evaluating.name],
    response_class=StreamingResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Sanity check the model being served with a trainer export",
)
def get_sanity_check_with_trainer_export(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> StreamingResponse:
    """
    Performs a sanity check on the running model using the provided trainer export files.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be uploaded.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A CSV file containing the sanity check results.
    """

    files = []
    file_names = []
    for te in trainer_export:
        temp_te = tempfile.NamedTemporaryFile()
        for line in te.file:
            temp_te.write(line)
        temp_te.flush()
        files.append(temp_te)
        file_names.append("" if te.filename is None else te.filename)
    try:
        concatenated = concat_trainer_exports([file.name for file in files], allow_recurring_doc_ids=False)
    finally:
        for file in files:
            file.close()
    concatenated = cast(Dict[str, Any], concatenated)
    concatenated = filter_by_concept_ids(concatenated, model_service.info().model_type)
    metrics = sanity_check_model_with_trainer_export(concatenated, model_service, return_df=True, include_anchors=False)
    metrics = cast(pd.DataFrame, metrics)
    stream = io.StringIO()
    metrics.to_csv(stream, index=False)
    tracking_id = tracking_id or str(uuid.uuid4())
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment ; filename="sanity_check_{tracking_id}.csv"'
    return response


@router.post(
    "/iaa-scores",
    tags=[Tags.Evaluating.name],
    response_class=StreamingResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Calculate inter annotator agreement scores between two projects",
)
def get_inter_annotator_agreement_scores(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="A list of trainer export files to be uploaded")],
    annotator_a_project_id: Annotated[int, Query(description="The project ID from one annotator")],
    annotator_b_project_id: Annotated[int, Query(description="The project ID from another annotator")],
    scope: Annotated[str, Query(enum=[s.value for s in Scope], description="The scope for which the score will be calculated, e.g., per_concept, per_document or per_span")],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
) -> StreamingResponse:
    """
    Calculates Inter-Annotator Agreement (IAA) scores between projects done by two annotators.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be uploaded.
        annotator_a_project_id (int): The project ID from the first annotator.
        annotator_b_project_id (int): The project ID from the second annotator.
        scope (str): The scope for which the IAA score will be calculated (per_concept, per_document, per_span).
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.

    Returns:
        StreamingResponse: A CSV file containing the IAA scores.

    Raises:
        AnnotationException: If the scope is not one of per_concept, per_document and per_span.
    """

    files = []
    for te in trainer_export:
        temp_te = tempfile.NamedTemporaryFile()
        for line in te.file:
            temp_te.write(line)
        temp_te.flush()
        files.append(temp_te)
    concatenated = concat_trainer_exports([file.name for file in files])
    for file in files:
        file.close()
    with tempfile.NamedTemporaryFile(mode="w+") as combined:
        json.dump(concatenated, combined)
        combined.seek(0)
        if scope == Scope.PER_CONCEPT.value:
            iaa_scores = get_iaa_scores_per_concept(combined, annotator_a_project_id, annotator_b_project_id, return_df=True)
        elif scope == Scope.PER_DOCUMENT.value:
            iaa_scores = get_iaa_scores_per_doc(combined, annotator_a_project_id, annotator_b_project_id, return_df=True)
        elif scope == Scope.PER_SPAN.value:
            iaa_scores = get_iaa_scores_per_span(combined, annotator_a_project_id, annotator_b_project_id, return_df=True)
        else:
            raise AnnotationException(f'Unknown scope: "{scope}"')
        iaa_scores = cast(pd.DataFrame, iaa_scores)
        stream = io.StringIO()
        iaa_scores.to_csv(stream, index=False)
        tracking_id = tracking_id or str(uuid.uuid4())
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment ; filename="iaa_{tracking_id}.csv"'
        return response


@router.post(
    "/concat_trainer_exports",
    tags=[Tags.Evaluating.name],
    response_class=JSONResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Concatenate multiple trainer export files into a single file for download",
)
def get_concatenated_trainer_exports(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="A list of trainer export files to be concatenated")],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
) -> JSONResponse:
    """
    Concatenates multiple trainer export files into a single file.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be concatenated.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.

    Returns:
        JSONResponse: A JSON file containing the combined trainer exports.
    """

    files = []
    for te in trainer_export:
        temp_te = tempfile.NamedTemporaryFile()
        for line in te.file:
            temp_te.write(line)
        temp_te.flush()
        files.append(temp_te)
    concatenated = concat_trainer_exports([file.name for file in files], allow_recurring_doc_ids=False)
    for file in files:
        file.close()
    tracking_id = tracking_id or str(uuid.uuid4())
    response = JSONResponse(concatenated, media_type="application/json; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment ; filename="concatenated_{tracking_id}.json"'
    return response


@router.post(
    "/annotation-stats",
    tags=[Tags.Evaluating.name],
    response_class=StreamingResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Get annotation stats of trainer export files",
)
def get_annotation_stats(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
) -> StreamingResponse:
    """
    Gets annotation statistics from the provided trainer export files.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be uploaded.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.

    Returns:
        StreamingResponse: A CSV file containing the annotation statistics.
    """

    files = []
    file_names = []
    for te in trainer_export:
        temp_te = tempfile.NamedTemporaryFile()
        for line in te.file:
            temp_te.write(line)
        temp_te.flush()
        files.append(temp_te)
        file_names.append("" if te.filename is None else te.filename)
    try:
        concatenated = concat_trainer_exports([file.name for file in files], allow_recurring_doc_ids=False)
    finally:
        for file in files:
            file.close()
    stats = get_stats_from_trainer_export(concatenated, return_df=True)
    stats = cast(pd.DataFrame, stats)
    stream = io.StringIO()
    stats.to_csv(stream, index=False)
    tracking_id = tracking_id or str(uuid.uuid4())
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment ; filename="stats_{tracking_id}.csv"'
    return response
