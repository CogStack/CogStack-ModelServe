import json
import tempfile
import uuid
import ijson
import logging
from typing import List, Union
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, UploadFile, Query, Request, File
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import api.globals as cms_globals
from domain import Tags
from model_services.base import AbstractModelService

router = APIRouter()
logger = logging.getLogger("cms")


@router.post("/train_unsupervised",
             status_code=HTTP_202_ACCEPTED,
             response_class=JSONResponse,
             tags=[Tags.Training.name],
             dependencies=[Depends(cms_globals.props.current_active_user)])
async def train_unsupervised(request: Request,
                             training_data: Annotated[List[UploadFile], File(description="One or more files to be uploaded and each contains a list of plain texts, in the format of [\"text_1\", \"text_2\", ..., \"text_n\"]")],
                             log_frequency: Annotated[int, Query(description="The number of processed documents after which training metrics will be logged", ge=1)] = 1000,
                             description: Annotated[Union[str, None], Query(description="The description of the training or change logs")] = None,
                             model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> JSONResponse:
    """
    Upload one or more plain text files and trigger the unsupervised training
    """
    data_file = tempfile.NamedTemporaryFile(mode="r+")
    files = []
    file_names = []
    data_file.write("[")
    for td_idx, td in enumerate(training_data):
        temp_td = tempfile.NamedTemporaryFile(mode="w")
        items = ijson.items(td.file, "item")
        temp_td.write("[")
        for text_idx, text in enumerate(items):
            if text_idx > 0 or td_idx > 0:
                data_file.write(",")
            json.dump(text, data_file)
            if text_idx > 0:
                temp_td.write(",")
            json.dump(text, temp_td)
        temp_td.write("]")
        temp_td.flush()
        file_names.append("" if td.filename is None else td.filename)
        files.append(temp_td)
    data_file.write("]")
    logger.debug("Training data concatenated")
    data_file.flush()
    data_file.seek(0)
    training_id = str(uuid.uuid4())
    try:
        training_accepted = model_service.train_unsupervised(data_file,
                                                             1,
                                                             log_frequency,
                                                             training_id,
                                                             ",".join(file_names),
                                                             raw_data_files=files,
                                                             synchronised=False,
                                                             description=description)
    finally:
        for file in files:
            file.close()

    return _get_training_response(training_accepted, training_id)


def _get_training_response(training_accepted: bool, training_id: str) -> JSONResponse:
    if training_accepted:
        logger.debug("Training accepted with ID: %s", training_id)
        return JSONResponse(content={"message": "Your training started successfully.", "training_id": training_id}, status_code=HTTP_202_ACCEPTED)
    else:
        logger.debug("Training refused due to another active training or evaluation on this model")
        return JSONResponse(content={"message": "Another training or evaluation on this model is still active. Please retry later."}, status_code=HTTP_503_SERVICE_UNAVAILABLE)
