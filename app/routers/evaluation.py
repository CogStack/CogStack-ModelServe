import io
import uuid

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import StreamingResponse

import globals
from domain import Tags
from model_services.base import AbstractModelService
from processors.metrics_collector import evaluate_model_with_trainer_export

router = APIRouter()


@router.post("/evaluate",
             tags=[Tags.Evaluating.name],
             response_class=StreamingResponse)
async def evaluate_using_trainer_export(trainer_export: UploadFile,
                                        model_service: AbstractModelService = Depends(globals.model_service_dep)) -> StreamingResponse:
    metrics = evaluate_model_with_trainer_export(trainer_export,
                                                 model_service,
                                                 return_df=True,
                                                 include_anchors=False)
    stream = io.StringIO()
    metrics.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment ; filename="evaluation_{str(uuid.uuid4())}.csv"'
    return response
