import tempfile
import uuid
import json
import logging
from typing import List
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, UploadFile, Query, Request, File
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import api.globals as globals
from domain import Tags
from model_services.base import AbstractModelService
from api.auth.users import props
from processors.metrics_collector import concat_trainer_exports
from utils import filter_by_concept_ids

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/train_supervised",
             status_code=HTTP_202_ACCEPTED,
             response_class=JSONResponse,
             tags=[Tags.Training.name],
             dependencies=[Depends(props.current_active_user)],
             description="Upload one or more trainer export files and trigger the supervised training")
async def train_supervised(request: Request,
                           trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")],
                           epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
                           log_frequency: Annotated[int, Query(description="The number of processed documents after which training metrics will be logged", ge=1)] = 1,
                           model_service: AbstractModelService = Depends(globals.model_service_dep)) -> JSONResponse:
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
    data_file = tempfile.NamedTemporaryFile(mode="w")
    concatenated = filter_by_concept_ids(concatenated)
    json.dump(concatenated, data_file)
    data_file.flush()
    data_file.seek(0)
    training_id = str(uuid.uuid4())
    training_accepted = model_service.train_supervised(data_file, epochs, log_frequency, training_id, ",".join(file_names))
    return _get_training_response(training_accepted, training_id)


def _get_training_response(training_accepted: bool, training_id: str) -> JSONResponse:
    if training_accepted:
        return JSONResponse(content={"message": "Your training started successfully.", "training_id": training_id}, status_code=HTTP_202_ACCEPTED)
    else:
        return JSONResponse(content={"message": "Another training on this model is still active. Please retry your training later."}, status_code=HTTP_503_SERVICE_UNAVAILABLE)
