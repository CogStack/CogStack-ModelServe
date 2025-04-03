import os
import tempfile
import uuid
import json
import logging
from typing import List, Tuple, Union
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, UploadFile, Query, Request, File, Form
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import app.api.globals as cms_globals
from app.api.dependencies import validate_tracking_id
from app.domain import Tags
from app.model_services.base import AbstractModelService
from app.processors.metrics_collector import concat_trainer_exports

router = APIRouter()
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.post(
    "/train_metacat",
    status_code=HTTP_202_ACCEPTED,
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Upload one or more trainer export files and trigger the metacat training",
)
async def train_metacat(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")],
    epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
    log_frequency: Annotated[int, Query(description="The number of processed documents or epochs after which training metrics will be logged", ge=1)] = 1,
    description: Annotated[Union[str, None], Form(description="The description on the training or change logs")] = None,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Triggers the Metacat training by uploading one or more trainer export files.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be uploaded for training.
        epochs (int, optional): The number of training epochs. Defaults to 1.
        log_frequency (int, optional): The number of processed documents or epochs after which training metrics will be logged. Defaults to 1.
        description (Union[str, None], optional): An optional description on the training or change logs.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response containing training response with the training ID.
    """

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
        logger.debug("Training exports concatenated")
    finally:
        for file in files:
            file.close()
    data_file = tempfile.NamedTemporaryFile("w")
    json.dump(concatenated, data_file)
    data_file.flush()
    data_file.seek(0)
    training_id = tracking_id or str(uuid.uuid4())
    try:
        training_response = model_service.train_metacat(
            data_file,
            epochs,
            log_frequency,
            training_id,
            ",".join(file_names),
            raw_data_files=files,
            synchronised=(os.environ.get("CMS_CI", "false") == "true"),
            description=description,
        )
    finally:
        for file in files:
            file.close()

    return _get_training_response(training_response, training_id)


def _get_training_response(training_response: Tuple[bool, str, str], training_id: str) -> JSONResponse:
    training_accepted, experiment_id, run_id = training_response
    if training_accepted:
        logger.debug("Training accepted with ID: %s", training_id)
        return JSONResponse(
            content={
                "message": "Your training started successfully.",
                "training_id": training_id,
                "experiment_id": experiment_id,
                "run_id": run_id,
            }, status_code=HTTP_202_ACCEPTED
        )
    else:
        logger.debug("Training refused due to another active training or evaluation on this model")
        return JSONResponse(
            content={
                "message": "Another training or evaluation on this model is still active. Please retry your training later.",
                "experiment_id": experiment_id,
                "run_id": run_id,
            }, status_code=HTTP_503_SERVICE_UNAVAILABLE
        )
