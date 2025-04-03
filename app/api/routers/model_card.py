import app.api.globals as cms_globals

from fastapi import APIRouter, Depends, Request
from app.domain import ModelCard, Tags
from app.model_services.base import AbstractModelService
from app.utils import get_settings
from app.api.utils import get_rate_limiter

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.get(
    "/info",
    response_model=ModelCard,
    tags=[Tags.Metadata.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Get information about the model being served",
)
async def get_model_card(
    request: Request,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> ModelCard:
    """
    Retrieves information about the model being served.

    Args:
        request (Request): The request object.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        ModelCard: A model card containing information about the model being served.
    """

    return model_service.info()
