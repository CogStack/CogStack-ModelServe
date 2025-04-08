import json
import logging
import sys
import tempfile
import uuid
from typing import List, Union, Dict, Any, cast
from typing_extensions import Annotated
from fastapi import APIRouter, Depends, Request, Query, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_202_ACCEPTED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from app.domain import Tags
from app.processors.metrics_collector import concat_trainer_exports
from app.utils import filter_by_concept_ids

import app.api.globals as cms_globals
from app.api.dependencies import validate_tracking_id
from app.model_services.base import AbstractModelService

router = APIRouter()
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.get(
    "/train_eval_info",
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Get the training or evaluation job information by its ID",
)
async def train_eval_info(
    request: Request,
    train_eval_id: Annotated[str, Query(description="The training or evaluation ID")],
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Retrieves the information of a training or evaluation job by its ID.

    Args:
        request (Request): The request object.
        train_eval_id (str): The ID of the training or evaluation job.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response containing the job information.
    """

    tracker_client = model_service.get_tracker_client()
    if tracker_client is None:
        return JSONResponse(status_code=HTTP_503_SERVICE_UNAVAILABLE,
                            content={"message": "The running model does not have any available trainers enabled"})
    infos = tracker_client.get_info_by_job_id(train_eval_id)
    return JSONResponse(status_code=HTTP_200_OK if len(infos) != 0 else HTTP_404_NOT_FOUND, content=infos)


@router.post(
    "/evaluate",
    tags=[Tags.Training.name],
    response_class=JSONResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Evaluate the model being served with a trainer export",
)
async def get_evaluation_with_trainer_export(
    request: Request,
    trainer_export: Annotated[List[UploadFile], File(description="One or more trainer export files to be uploaded")],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Evaluates the running model with one or more trainer export files.

    Args:
        request (Request): The request object.
        trainer_export (List[UploadFile]): A list of trainer export files to be uploaded.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        SONResponse: A JSON response containing training response with the evaluation ID.
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
    finally:
        for file in files:
            file.close()
    data_file = tempfile.NamedTemporaryFile(mode="w")
    concatenated = filter_by_concept_ids(cast(Dict[str, Any], concatenated), model_service.info().model_type)
    json.dump(concatenated, data_file)
    data_file.flush()
    data_file.seek(0)
    evaluation_id = tracking_id or str(uuid.uuid4())
    evaluation_accepted, experiment_id, run_id = model_service.train_supervised(
        data_file, 0, sys.maxsize, evaluation_id, ",".join(file_names)
    )
    if evaluation_accepted:
        return JSONResponse(
            content={
                "message": "Your evaluation started successfully.",
                "evaluation_id": evaluation_id,
                "experiment_id": experiment_id,
                "run_id": run_id,
            }, status_code=HTTP_202_ACCEPTED
        )
    else:
        return JSONResponse(
            content={
                "message": "Another training or evaluation on this model is still active. Please retry later.",
                "experiment_id": experiment_id,
                "run_id": run_id,
            }, status_code=HTTP_503_SERVICE_UNAVAILABLE
        )


@router.post(
    "/cancel_training",
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Cancel the in-progress training job (this is experimental and may not work as expected)",
)
async def cancel_training(
    request: Request,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Cancels the in-progress training job (Experimental, the support for this may vary depending on the underlying model).

    Args:
        request (Request): The request object.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response indicating whether the cancellation was accepted or not.
    """

    training_cancelled = model_service.cancel_training()
    if not training_cancelled:
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"message": "Cannot find in-progress training or no trainers are enabled for the running model."},
        )
    return JSONResponse(
        status_code=HTTP_202_ACCEPTED,
        content={"message": "The in-progress training will be stopped momentarily."},
    )


@router.get(
    "/train_eval_metrics",
    response_class=JSONResponse,
    tags=[Tags.Training.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Get the training or evaluation metrics by its ID (Each metric may contain multiple values for multiple epochs)",
)
async def train_eval_metrics(
    request: Request,
    train_eval_id: Annotated[str, Query(description="The training or evaluation ID")],
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Retrieves the metrics of a training or evaluation job by its ID.

    Args:
        request (Request): The request object.
        train_eval_id (str): The training or evaluation ID.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response containing the training or evaluation metrics.
    """

    tracker_client = model_service.get_tracker_client()
    if tracker_client is None:
        return JSONResponse(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            content={"message": "The running model does not have any available trainers enabled"},
        )
    metrics = tracker_client.get_metrics_by_job_id(train_eval_id)
    return JSONResponse(status_code=HTTP_200_OK if len(metrics) != 0 else HTTP_404_NOT_FOUND, content=metrics)
