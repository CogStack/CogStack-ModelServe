import logging
from typing_extensions import Annotated
from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import JSONResponse
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND, HTTP_503_SERVICE_UNAVAILABLE
from domain import Tags

import api.globals as cms_globals
from model_services.base import AbstractModelService

router = APIRouter()
logger = logging.getLogger("cms")

@router.get("/train_info",
            response_class=JSONResponse,
            tags=[Tags.Training.name],
            dependencies=[Depends(cms_globals.props.current_active_user)],
            description="Get the training information by a given training ID")
def training_info(request: Request,
                  training_id: Annotated[str, Query(description="Training ID")],
                  model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> JSONResponse:
    tracker_client = model_service.get_tracker_client()
    if tracker_client is None:
        return JSONResponse(status_code=HTTP_503_SERVICE_UNAVAILABLE,
                            content={"message": "The running model does not have any available trainers enabled"})
    infos = tracker_client.get_info_by_job_id(training_id)
    return JSONResponse(status_code=HTTP_200_OK if len(infos) != 0 else HTTP_404_NOT_FOUND, content=infos)
