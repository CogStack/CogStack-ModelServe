import io
import json
import uuid
import tempfile
import logging

from typing import List
from typing_extensions import Annotated
from fastapi import APIRouter, Query, Depends, UploadFile, Request, File
from fastapi.responses import StreamingResponse, JSONResponse

import api.globals as globals
from domain import Tags, Scope
from model_services.base import AbstractModelService
from processors.metrics_collector import (
    evaluate_model_with_trainer_export,
    get_iaa_scores_per_concept,
    get_iaa_scores_per_doc,
    get_iaa_scores_per_span,
    concat_trainer_exports,
)
from api.auth.users import props
from exception import AnnotationException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/evaluate",
             tags=[Tags.Evaluating.name],
             response_class=StreamingResponse,
             dependencies=[Depends(props.current_active_user)],
             description="Evaluate a trainer export against the current served model")
def get_evaluation_with_trainer_export(request: Request,
                                       trainer_export: Annotated[UploadFile, File(description="The trainer export file to be uploaded")],
                                       model_service: AbstractModelService = Depends(globals.model_service_dep)) -> StreamingResponse:
    with tempfile.NamedTemporaryFile() as file:
        for line in trainer_export.file:
            file.write(line)
        file.seek(0)
        metrics = evaluate_model_with_trainer_export(file,
                                                     model_service,
                                                     return_df=True,
                                                     include_anchors=False)
        stream = io.StringIO()
        metrics.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment ; filename="evaluation_{str(uuid.uuid4())}.csv"'
        return response


@router.post("/iaa-scores",
             tags=[Tags.Evaluating.name],
             response_class=StreamingResponse,
             dependencies=[Depends(props.current_active_user)],
             description="Calculate inter annotator agreement scores between two projects")
def get_inter_annotator_agreement_scores(request: Request,
                                         trainer_export: Annotated[List[UploadFile], File(description="A list of trainer export files to be uploaded")],
                                         annotator_a_project_id: Annotated[int, Query(description="The project ID from one annotator")],
                                         annotator_b_project_id: Annotated[int, Query(description="The project ID from another annotator")],
                                         scope: Annotated[str, Query(enum=[s.value for s in Scope], description="The scope for which the score will be calculated, e.g., per_concept, per_document or per_span")]) -> StreamingResponse:
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
        stream = io.StringIO()
        iaa_scores.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment ; filename="evaluation_{str(uuid.uuid4())}.csv"'
        return response


@router.post("/concat_trainer_exports",
             tags=[Tags.Evaluating.name],
             response_class=JSONResponse,
             dependencies=[Depends(props.current_active_user)],
             description="Concatenate multiple trainer export files into a single file for download")
def get_concatenated_trainer_exports(request: Request,
                                     trainer_export: Annotated[List[UploadFile], File(description="A list of trainer export files to be uploaded")]) -> JSONResponse:
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
    response = JSONResponse(concatenated, media_type="application/json; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment ; filename="concatenated_{str(uuid.uuid4())}.json"'
    return response
