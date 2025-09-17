import os
import tempfile
import uuid
import json
import logging
from typing import List, Tuple, Union, Any, Dict, cast
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, UploadFile, Query, Request, File, Form
from fastapi.responses import JSONResponse
from starlette.status import HTTP_202_ACCEPTED, HTTP_503_SERVICE_UNAVAILABLE

import app.api.globals as cms_globals
from app.api.dependencies import validate_tracking_id
from app.domain import Tags, ModelType
from app.model_services.base import AbstractModelService
from app.processors.metrics_collector import concat_json_lists, concat_trainer_exports
from app.utils import filter_by_concept_ids

router = APIRouter()
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.post(
    "/train_supervised",
    status_code=HTTP_202_ACCEPTED,
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Upload one or more trainer export files and trigger the supervised training",
)
async def train_supervised(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")],
    epochs: Annotated[int, Query(description="The number of training epochs", ge=0)] = 1,
    lr_override: Annotated[Union[float, None], Query(description="The override of the initial learning rate", gt=0.0)] = None,
    test_size: Annotated[Union[float, None], Query(description="The override of the test size in percentage. (For a 'huggingface-ner' model, a negative value can be used to apply the train-validation-test split if implicitly defined in trainer export: 'projects[0]' is used for training, 'projects[1]' for validation, and 'projects[2]' for testing)")] = 0.2,
    early_stopping_patience: Annotated[Union[int, None], Query(description="The number of evaluations to wait for improvement before stopping the training. (Non-positive values disable early stopping)")] = -1,
    log_frequency: Annotated[int, Query(description="The number of processed documents or epochs after which training metrics will be logged", ge=1)] = 1,
    description: Annotated[Union[str, None], Form(description="The description of the training or change logs")] = None,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Triggers supervised training on the running model using uploaded trainer export files.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be uploaded.
        epochs (int): The number of training epochs. Defaults to 1.
        lr_override (float, optional): The override of the initial learning rate. Defaults to the value used in previous training and must be greater than 0.0.
        test_size (float, optional): The override of the test size in percentage. Defaults to 0.2 and can be negative for 'huggingface-ner' models to apply the train-validation-test split if implicitly defined in trainer export: 'projects[0]' is used for training, 'projects[1]' for validation, and 'projects[2]' for testing.
        early_stopping_patience (int, optional): The number of evaluations to wait for improvement before stopping the training. Non-positive values disable early stopping.
        log_frequency (int): The number of processed documents or epochs after which training metrics will be logged. Must be at least 1.
        description (str, optional): The description of the training or change logs. Defaults to empty.
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

    if model_service.info().model_type is not ModelType.HUGGINGFACE_LLM:
        concatenated_te = concat_trainer_exports([file.name for file in files], allow_recurring_doc_ids=False)
        logger.debug("Training exports concatenated")
        data_file = tempfile.NamedTemporaryFile(mode="w+")
        concatenated_te = filter_by_concept_ids(cast(Dict[str, Any], concatenated_te), model_service.info().model_type)
        logger.debug("Training exports filtered by concept IDs")
        json.dump(concatenated_te, data_file)
    else:
        concatenated = concat_json_lists([file.name for file in files])
        logger.debug("Training exports concatenated")
        data_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(concatenated, data_file)

    data_file.flush()
    data_file.seek(0)
    training_id = tracking_id or str(uuid.uuid4())
    try:
        training_response = model_service.train_supervised(
            data_file,
            epochs,
            log_frequency,
            training_id,
            ",".join(file_names),
            raw_data_files=files,
            description=description,
            synchronised=(os.environ.get("CMS_CI", "false") == "true"),
            lr_override=lr_override,
            test_size=test_size,
            early_stopping_patience=early_stopping_patience,
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
