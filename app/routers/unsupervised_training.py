import json
import tempfile
import uuid
import ijson
from typing import List

from fastapi import APIRouter, Depends, UploadFile, Query, Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import globals
from domain import Tags
from model_services.base import AbstractModelService
from auth.users import props

router = APIRouter()


@router.post("/train_unsupervised",
             status_code=HTTP_202_ACCEPTED,
             response_class=JSONResponse,
             tags=[Tags.Training.name],
             dependencies=[Depends(props.current_active_user)])
async def train_unsupervised(request: Request,
                             training_data: List[UploadFile],
                             log_frequency: int = Query(default=1000, description="log after every number of processed documents", ge=1),
                             model_service: AbstractModelService = Depends(globals.model_service_dep)) -> JSONResponse:
    """
    Upload one or more plain text files and trigger the unsupervised training

    - **training_data**: one or more files to be uploaded and each contains a list of plain texts, in the format of ["text_1", "text_2", ..., "text_n"]
    - **log_frequency**: the number of processed documents after which training metrics will be logged
    """
    data_file = tempfile.NamedTemporaryFile(mode="r+")
    file_names = []
    data_file.write("[")
    for td_idx, td in enumerate(training_data):
        items = ijson.items(td.file, "item")
        for text_idx, text in enumerate(items):
            if text_idx > 0 or td_idx > 0:
                data_file.write(",")
            json.dump(text, data_file)
        file_names.append("" if td.filename is None else td.filename)
    data_file.write("]")
    data_file.flush()
    data_file.seek(0)
    training_id = str(uuid.uuid4())
    training_accepted = model_service.train_unsupervised(data_file, 1, log_frequency, training_id, ",".join(file_names))
    return _get_training_response(training_accepted, training_id)


def _get_training_response(training_accepted: bool, training_id: str) -> JSONResponse:
    if training_accepted:
        return JSONResponse(content={"message": "Your training started successfully.", "training_id": training_id}, status_code=HTTP_202_ACCEPTED)
    else:
        return JSONResponse(content={"message": "Another training on this model is still active. Please retry your training later."}, status_code=HTTP_503_SERVICE_UNAVAILABLE)
