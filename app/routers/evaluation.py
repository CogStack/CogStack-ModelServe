import io
import uuid
import tempfile

from starlette.status import HTTP_404_NOT_FOUND
from fastapi import APIRouter, Depends, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

import globals
from domain import Tags
from model_services.base import AbstractModelService
from processors.metrics_collector import evaluate_model_with_trainer_export, get_intra_annotator_agreement_scores

router = APIRouter()


@router.post("/evaluate",
             tags=[Tags.Evaluating.name],
             response_class=StreamingResponse)
async def evaluate_using_trainer_export(trainer_export: UploadFile,
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
             response_class=StreamingResponse)
async def intra_annotator_agreement_scores(trainer_export: UploadFile,
                                           annotator_a_project_id: int,
                                           annotator_b_project_id: int) -> StreamingResponse:
    with tempfile.NamedTemporaryFile() as file:
        for line in trainer_export.file:
            file.write(line)
        file.seek(0)
        try:
            iaa_scores = get_intra_annotator_agreement_scores(file, annotator_a_project_id, annotator_b_project_id, return_df=True)
        except ValueError as e:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail=str(e))
        stream = io.StringIO()
        iaa_scores.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment ; filename="evaluation_{str(uuid.uuid4())}.csv"'
        return response
