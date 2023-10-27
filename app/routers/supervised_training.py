import tempfile
import uuid
import json
import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, Response, UploadFile, Query, HTTPException
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE, HTTP_400_BAD_REQUEST

import globals
from domain import Tags
from model_services.base import AbstractModelService
from auth.users import props
from processors.metrics_collector import concat_trainer_exports
from utils import filter_by_concept_ids
from exception import AnnotationException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/train_supervised",
             status_code=HTTP_202_ACCEPTED,
             tags=[Tags.Training.name],
             dependencies=[Depends(props.current_active_user)])
async def train_supervised(trainer_export: List[UploadFile],
                           response: Response,
                           epochs: int = Query(default=1, description="The number of training epochs", ge=0),
                           log_frequency: int = Query(default=1, description="log after every number of finished epochs", ge=1),
                           model_service: AbstractModelService = Depends(globals.model_service_dep)) -> Dict:
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
    except AnnotationException as e:
        logger.exception(e)
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))
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
    return _get_training_response(training_accepted, response, training_id)


def _get_training_response(training_accepted: bool, response: Response, training_id: str) -> Dict:
    if training_accepted:
        return {"message": "Your training started successfully.", "training_id": training_id}
    else:
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE
        return {"message": "Another training on this model is still active. Please retry your training later."}
