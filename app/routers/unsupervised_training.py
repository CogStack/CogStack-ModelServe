import tempfile
import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends, Response, UploadFile, Query
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import globals
from domain import Tags
from model_services.base import AbstractModelService
from auth.users import props

router = APIRouter()


@router.post("/train_unsupervised",
             status_code=HTTP_202_ACCEPTED,
             tags=[Tags.Training.name],
             dependencies=[Depends(props.current_active_user)])
async def train_unsupervised(response: Response,
                             training_data: List[UploadFile],
                             log_frequency: int = Query(default=1000, description="log after every number of processed documents", ge=1),
                             model_service: AbstractModelService = Depends(globals.model_service_dep)) -> Dict:
    data_file = tempfile.NamedTemporaryFile()
    file_names = []
    for td in training_data:
        for line in td.file:
            data_file.write(line)
        file_names.append("" if td.filename is None else td.filename)
    data_file.flush()
    data_file.seek(0)
    training_id = str(uuid.uuid4())
    training_accepted = model_service.train_unsupervised(data_file, 1, log_frequency, training_id, ",".join(file_names))
    return _get_training_response(training_accepted, response, training_id)


def _get_training_response(training_accepted: bool, response: Response, training_id: str) -> Dict:
    if training_accepted:
        return {"message": "Your training started successfully.", "training_id": training_id}
    else:
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE
        return {"message": "Another training on this model is still active. Please retry your training later."}
