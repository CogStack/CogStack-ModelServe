import logging
import app.api.globals as cms_globals

from typing_extensions import Annotated
from fastapi import APIRouter, Depends, Request, Body, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from app.domain import Tags
from app.model_services.base import AbstractModelService
from app.utils import get_settings
from app.api.utils import get_rate_limiter

PATH_GENERATE = "/generate"
PATH_GENERATE_ASYNC = "/stream/generate"

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.post(
    PATH_GENERATE,
    tags=[Tags.Generative.name],
    response_class=PlainTextResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate text",
)
def generate_text(
    request: Request,
    prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
    max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> PlainTextResponse:
    """
    Generate text based on the prompt provided.

    Args:
        request (Request): The request object.
        prompt (str): The prompt to be sent to the model.
        max_tokens (int): The maximum number of tokens to generate.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        PlainTextResponse: A response containing the generated text.
    """

    return PlainTextResponse(model_service.generate(prompt, max_tokens=max_tokens))


@router.post(
    PATH_GENERATE_ASYNC,
    tags=[Tags.Generative.name],
    response_class=StreamingResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate a stream of texts",
)
async def generate_text_stream(
    request: Request,
    prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
    max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> StreamingResponse:
    """
    Generate a stream of texts in near real-time.

    Args:
        request (Request): The request object.
        prompt (str): The prompt to be sent to the model.
        max_tokens (int): The maximum number of tokens to generate.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A streaming response containing the text generated in near real-time.
    """

    return StreamingResponse(
        model_service.generate_async(prompt, max_tokens=max_tokens),
        media_type="text/event-stream"
    )
