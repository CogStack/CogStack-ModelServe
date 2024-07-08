import api.globals as cms_globals

from fastapi import APIRouter, Depends, Request
from domain import ModelCard, Tags
from model_services.base import AbstractModelService
from utils import get_settings
from api.utils import get_rate_limiter

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)


@router.get("/info",
            response_model=ModelCard,
            tags=[Tags.Metadata.name],
            dependencies=[Depends(cms_globals.props.current_active_user)],
            description="Get information about the model being served")
async def get_model_card(request: Request,
                         model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> ModelCard:
    return model_service.info()
