import tempfile
import uuid
from typing import Dict

from fastapi import APIRouter, Depends, Response, UploadFile, Query
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import globals
from domain import Tags
from model_services.base import AbstractModelService

router = APIRouter()


@router.post("/train_supervised", status_code=HTTP_202_ACCEPTED, tags=[Tags.Training.name])
async def supervised_training(training_data: UploadFile,
                              response: Response,
                              epochs: int = 1,
                              log_frequency: int = Query(default=1, description="log after every number of finished epochs"),
                              model_service: AbstractModelService = Depends(globals.model_service_dep)) -> Dict:
    data_file = tempfile.NamedTemporaryFile()
    for line in training_data.file:
        data_file.write(line)
    data_file.flush()
    training_id = str(uuid.uuid4())
    training_accepted = model_service.train_supervised(data_file, epochs, log_frequency, training_id, training_data.filename)
    return _get_training_response(training_accepted, response, training_id)


def _get_training_response(training_accepted: bool, response: Response, training_id: str) -> Dict:
    if training_accepted:
        return {"message": "Your training started successfully.", "training_id": training_id}
    else:
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE
        return {"message": "Another training is in progress. Please retry your training later."}
