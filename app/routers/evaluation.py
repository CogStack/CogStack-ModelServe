import io
import json
import uuid
import tempfile
import logging

from typing import List
from starlette.status import HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST
from fastapi import APIRouter, Query, Depends, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

import globals
from domain import Tags, Scope
from model_services.base import AbstractModelService
from processors.metrics_collector import (
    evaluate_model_with_trainer_export,
    get_iaa_scores_per_concept,
    get_iaa_scores_per_doc,
    get_iaa_scores_per_span,
    concat_trainer_exports,
)
from auth.users import props

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/evaluate",
             tags=[Tags.Evaluating.name],
             response_class=StreamingResponse,
             dependencies=[Depends(props.current_active_user)])
def get_evaluation_with_trainer_export(trainer_export: UploadFile,
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
             dependencies=[Depends(props.current_active_user)])
def get_intra_annotator_agreement_scores(trainer_export: List[UploadFile],
                                         annotator_a_project_id: int,
                                         annotator_b_project_id: int,
                                         scope: str = Query("scope", enum=[s.value for s in Scope])) -> StreamingResponse:
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
        try:
            if scope == Scope.PER_CONCEPT.value:
                iaa_scores = get_iaa_scores_per_concept(combined, annotator_a_project_id, annotator_b_project_id, return_df=True)
            elif scope == Scope.PER_DOCUMENT.value:
                iaa_scores = get_iaa_scores_per_doc(combined, annotator_a_project_id, annotator_b_project_id, return_df=True)
            elif scope == Scope.PER_SPAN.value:
                iaa_scores = get_iaa_scores_per_span(combined, annotator_a_project_id, annotator_b_project_id, return_df=True)
            else:
                raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f'Unknown scope: "{scope}"')
        except ValueError as e:
            logger.exception(e)
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(e))
        stream = io.StringIO()
        iaa_scores.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment ; filename="evaluation_{str(uuid.uuid4())}.csv"'
        return response


@router.post("/concat_trainer_exports",
             tags=[Tags.Evaluating.name],
             response_class=JSONResponse,
             dependencies=[Depends(props.current_active_user)])
def get_concatenated_trainer_exports(trainer_export: List[UploadFile]) -> JSONResponse:
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
    response = JSONResponse(concatenated, media_type="application/json; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment ; filename="concatenated_{str(uuid.uuid4())}.json"'
    return response
