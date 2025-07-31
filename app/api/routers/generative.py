import json
import logging
import time
import uuid
import app.api.globals as cms_globals

from typing import Union, Iterable, AsyncGenerator
from typing_extensions import Annotated
from functools import partial
from fastapi import APIRouter, Depends, Request, Body, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST
from app.domain import Tags, OpenAIChatRequest, OpenAIChatResponse, PromptMessage, PromptRole
from app.model_services.base import AbstractModelService
from app.utils import get_settings, get_prompt_from_messages
from app.api.utils import get_rate_limiter
from app.api.dependencies import validate_tracking_id
from app.management.prometheus_metrics import cms_prompt_tokens, cms_completion_tokens, cms_total_tokens

PATH_GENERATE = "/generate"
PATH_GENERATE_ASYNC = "/stream/generate"
PATH_OPENAI_COMPLETIONS = "/v1/chat/completions"

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
    temperature: Annotated[float, Query(description="The temperature of the generated text", gt=0.0, lt=1.0)] = 0.7,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> PlainTextResponse:
    """
    Generates text based on the prompt provided.

    Args:
        request (Request): The request object.
        prompt (str): The prompt to be sent to the model.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature of the generated text.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        PlainTextResponse: A response containing the generated text.
    """

    tracking_id = tracking_id or str(uuid.uuid4())
    if prompt:
        return PlainTextResponse(
            model_service.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                report_tokens=partial(_send_usage_metrics, handler=PATH_GENERATE),
            ),
            headers={"x-cms-tracking-id": tracking_id},
            status_code=HTTP_200_OK,
        )
    else:
        return PlainTextResponse(
            _empty_prompt_error(),
            headers={"x-cms-tracking-id": tracking_id},
            status_code=HTTP_400_BAD_REQUEST,
        )


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
    temperature: Annotated[float, Query(description="The temperature of the generated text", gt=0.0, lt=1.0)] = 0.7,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> StreamingResponse:
    """
    Generates a stream of texts in near real-time.

    Args:
        request (Request): The request object.
        prompt (str): The prompt to be sent to the model.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature of the generated text.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A streaming response containing the text generated in near real-time.
    """

    tracking_id = tracking_id or str(uuid.uuid4())
    if prompt:
        return StreamingResponse(
            model_service.generate_async(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                report_tokens=partial(_send_usage_metrics, handler=PATH_GENERATE_ASYNC),
            ),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
            status_code=HTTP_200_OK,
        )
    else:
        return StreamingResponse(
            _empty_prompt_error(),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
            status_code=HTTP_400_BAD_REQUEST,
        )


@router.post(
    PATH_OPENAI_COMPLETIONS,
    tags=[Tags.Generative.name],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate chat response based on messages, similar to OpenAI's /v1/chat/completions",
)
def generate_chat_completions(
    request: Request,
    request_data: Annotated[OpenAIChatRequest, Body(
        description="OpenAI-like completion request", media_type="application/json"
    )],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Generates chat response based on messages, mimicking OpenAI's /v1/chat/completions endpoint.

    Args:
        request (Request): The request object.
        request_data (OpenAIChatRequest): The request data containing model, messages, and stream.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A OpenAI-like response containing the text generated in near real-time.
        JSONResponse: A response containing an error message if the prompt messages are empty.
    """

    messages = request_data.messages
    stream = request_data.stream
    max_tokens = request_data.max_tokens
    temperature = request_data.temperature
    tracking_id = tracking_id or str(uuid.uuid4())

    if not messages:
        error_response = {
            "error": {
                "message": "No prompt messages provided",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "missing_field",
            }
        }
        return JSONResponse(
            content=error_response,
            status_code=HTTP_400_BAD_REQUEST,
            headers={"x-cms-tracking-id": tracking_id},
        )

    async def _stream(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator:
        data = {
            "id": tracking_id,
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"role": PromptRole.ASSISTANT.value}}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        async for chunk in model_service.generate_async(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            report_tokens=partial(_send_usage_metrics, handler=PATH_OPENAI_COMPLETIONS)
        ):
            data = {
                "choices": [
                    {
                        "delta": {"content": chunk}
                    }
                ],
                "object": "chat.completion.chunk",
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    prompt = get_prompt_from_messages(model_service.tokenizer, messages)    # type: ignore
    if stream:
        return StreamingResponse(
            _stream(prompt, max_tokens, temperature),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
        )
    else:
        generated_text = model_service.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            send_metrics=partial(_send_usage_metrics, handler=PATH_OPENAI_COMPLETIONS),
        )
        completion = OpenAIChatResponse(
            id=tracking_id,
            object="chat.completion",
            created=int(time.time()),
            model=model_service.model_name,
            choices=[
                {
                    "index": 0,
                    "message": PromptMessage(
                        role=PromptRole.ASSISTANT,
                        content=generated_text,
                    ),
                    "finish_reason": "stop",
                }
            ],
        )
        return JSONResponse(content=jsonable_encoder(completion), headers={"x-cms-tracking-id": tracking_id})


def _empty_prompt_error() -> Iterable[str]:
    yield "ERROR: No prompt text provided\n"


def _send_usage_metrics(handler: str, prompt_token_num: int, completion_token_num: int) -> None:
    cms_prompt_tokens.labels(handler=handler).observe(prompt_token_num)
    logger.debug(f"Sent prompt tokens usage: {prompt_token_num}")
    cms_completion_tokens.labels(handler=handler).observe(completion_token_num)
    logger.debug(f"Sent completion tokens usage: {completion_token_num}")
    cms_total_tokens.labels(handler=handler).observe(prompt_token_num + completion_token_num)
    logger.debug(f"Sent total tokens usage: {prompt_token_num + completion_token_num}")
