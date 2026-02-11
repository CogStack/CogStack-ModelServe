import json
import logging
import time
import uuid
import app.api.globals as cms_globals

from typing import Union, Iterable, AsyncGenerator, List
from typing_extensions import Annotated
from functools import partial
from fastapi import APIRouter, Depends, Request, Body, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from app.domain import (
    Tags,
    TagsGenerative,
    OpenAIChatCompletionsRequest,
    OpenAIChatCompletionsResponse,
    OpenAICompletionsRequest,
    OpenAICompletionsResponse,
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
    PromptMessage,
    PromptRole,
)
from app.model_services.base import AbstractModelService
from app.utils import get_settings, get_prompt_from_messages
from app.api.utils import get_rate_limiter
from app.api.dependencies import validate_tracking_id
from app.management.prometheus_metrics import cms_prompt_tokens, cms_completion_tokens, cms_total_tokens

PATH_GENERATE = "/generate"
PATH_GENERATE_ASYNC = "/stream/generate"
PATH_GENERATE_SSE = "/events/generate"
PATH_CHAT_COMPLETIONS = "/v1/chat/completions"
PATH_COMPLETIONS = "/v1/completions"
PATH_EMBEDDINGS = "/v1/embeddings"

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"

@router.post(
    PATH_GENERATE,
    tags=[TagsGenerative.Generative],
    response_class=PlainTextResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate text",
)
def generate_text(
    request: Request,
    prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
    max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512,
    temperature: Annotated[float, Query(description="The temperature of the generated text", ge=0.0)] = 0.7,
    top_p: Annotated[float, Query(description="The Top-P value for nucleus sampling", ge=0.0, le=1.0)] = 0.9,
    stop_sequences: Annotated[List[str], Query(description="The list of sequences used to stop the generation")] = [],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
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
        top_p (float): The Top-P value for nucleus sampling.
        stop_sequences (List[str]): The list of sequences used to stop the generation.
        ensure_full_sentences (bool): Whether to generate full sentences only.
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
                top_p=top_p,
                stop_sequences=stop_sequences,
                report_tokens=partial(_send_usage_metrics, handler=PATH_GENERATE),
                ensure_full_sentences=ensure_full_sentences,
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
    tags=[TagsGenerative.Generative],
    response_class=StreamingResponse,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate a stream of texts",
)
async def generate_text_stream(
    request: Request,
    prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
    max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512,
    temperature: Annotated[float, Query(description="The temperature of the generated text", ge=0.0)] = 0.7,
    top_p: Annotated[float, Query(description="The Top-P value for nucleus sampling", ge=0.0, le=1.0)] = 0.9,
    stop_sequences: Annotated[List[str], Query(description="The list of sequences used to stop the generation")] = [],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
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
        top_p (float): The Top-P value for nucleus sampling.
        stop_sequences (List[str]): The list of sequences used to stop the generation.
        ensure_full_sentences (bool): Whether to generate full sentences only.
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
                top_p=top_p,
                stop_sequences=stop_sequences,
                report_tokens=partial(_send_usage_metrics, handler=PATH_GENERATE_ASYNC),
                ensure_full_sentences=ensure_full_sentences,
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
    PATH_CHAT_COMPLETIONS,
    tags=[Tags.OpenAICompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate chat response based on messages, similar to OpenAI's /v1/chat/completions",
)
def generate_chat_completions(
    request: Request,
    request_data: Annotated[OpenAIChatCompletionsRequest, Body(
        description="OpenAI-like completion request", media_type="application/json"
    )],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Generates chat response based on messages, mimicking OpenAI's /v1/chat/completions endpoint.

    Args:
        request (Request): The request object.
        request_data (OpenAIChatRequest): The request data containing model, messages, stream, temperature, top_p, and stop_sequences.
        ensure_full_sentences (bool): Whether to generate full sentences only.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A OpenAI-like response containing the text generated in near real-time.
        JSONResponse: A response containing an error message if the prompt messages are empty.
    """

    messages = request_data.messages
    model = model_service.model_name if request_data.model != model_service.model_name else request_data.model
    stream = request_data.stream
    max_tokens = request_data.max_tokens
    temperature = request_data.temperature
    top_p = request_data.top_p
    stop_sequences = request_data.stop_sequences
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

    async def _stream(
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: List[str],
        ensure_full_sentences: bool,
    ) -> AsyncGenerator:
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
            top_p=top_p,
            stop_sequences=stop_sequences,
            report_tokens=partial(_send_usage_metrics, handler=PATH_CHAT_COMPLETIONS),
            ensure_full_sentences=ensure_full_sentences,
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

    assert hasattr(model_service, "tokenizer"), "Model service doesn't have a tokenizer"
    prompt = get_prompt_from_messages(model_service.tokenizer, messages)
    if stream:
        return StreamingResponse(
            _stream(prompt, max_tokens, temperature, top_p, stop_sequences or [], ensure_full_sentences),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
        )
    else:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        def _report_tokens(prompt_token_num: int, completion_token_num: int) -> None:
            usage["prompt_tokens"] = prompt_token_num
            usage["completion_tokens"] = completion_token_num
            usage["total_tokens"] = prompt_token_num + completion_token_num
            _send_usage_metrics(
                handler=PATH_CHAT_COMPLETIONS,
                prompt_token_num=prompt_token_num,
                completion_token_num=completion_token_num,
            )
        generated_text = model_service.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences or [],
            report_tokens=_report_tokens,
            ensure_full_sentences=ensure_full_sentences,
        )
        completion = OpenAIChatCompletionsResponse(
            id=tracking_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
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
            usage=usage,
        )
        return JSONResponse(content=jsonable_encoder(completion), headers={"x-cms-tracking-id": tracking_id})


@router.post(
    PATH_COMPLETIONS,
    tags=[Tags.OpenAICompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate completion based on prompt, similar to OpenAI's /v1/completions",
)
def generate_text_completions(
    request: Request,
    request_data: Annotated[OpenAICompletionsRequest, Body(
        description="OpenAI-like completion request", media_type="application/json"
    )],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Generates completion response based on prompt, mimicking OpenAI's /v1/completions endpoint.

    Args:
        request (Request): The request object.
        request_data (OpenAICompletionsRequest): The request data containing model, prompt, stream, temperature, top_p, and stop.
        ensure_full_sentences (bool): Whether to generate full sentences only.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: An OpenAI-like streaming response.
        JSONResponse: A response containing the generated text or an error message.
    """

    tracking_id = tracking_id or str(uuid.uuid4())
    model = model_service.model_name if request_data.model != model_service.model_name else request_data.model
    stream = request_data.stream
    max_tokens = request_data.max_tokens
    temperature = request_data.temperature
    top_p = request_data.top_p
    stop = request_data.stop

    if isinstance(stop, str):
        stop_sequences = [stop]
    elif isinstance(stop, list):
        stop_sequences = stop
    else:
        stop_sequences = []

    if isinstance(request_data.prompt, str):
        prompt = request_data.prompt
    else:
        prompt = "\n".join(request_data.prompt)

    if not prompt:
        error_response = {
            "error": {
                "message": "No prompt provided",
                "type": "invalid_request_error",
                "param": "prompt",
                "code": "missing_field",
            }
        }
        return JSONResponse(
            content=error_response,
            status_code=HTTP_400_BAD_REQUEST,
            headers={"x-cms-tracking-id": tracking_id},
        )

    async def _stream(
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: List[str],
        ensure_full_sentences: bool,
    ) -> AsyncGenerator:
        data = {
            "id": tracking_id,
            "object": "text_completion",
            "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        async for chunk in model_service.generate_async(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            report_tokens=partial(_send_usage_metrics, handler=PATH_COMPLETIONS),
            ensure_full_sentences=ensure_full_sentences,
        ):
            data = {
                "object": "text_completion",
                "choices": [
                    {
                        "text": chunk,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(
            _stream(prompt, max_tokens, temperature, top_p, stop_sequences, ensure_full_sentences),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
        )

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    def _report_tokens(prompt_token_num: int, completion_token_num: int) -> None:
        usage["prompt_tokens"] = prompt_token_num
        usage["completion_tokens"] = completion_token_num
        usage["total_tokens"] = prompt_token_num + completion_token_num
        _send_usage_metrics(
            handler=PATH_COMPLETIONS,
            prompt_token_num=prompt_token_num,
            completion_token_num=completion_token_num,
        )
    generated_text = model_service.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences,
        report_tokens=_report_tokens,
        ensure_full_sentences=ensure_full_sentences,
    )

    completion = OpenAICompletionsResponse(
        id=tracking_id,
        object="text_completion",
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "text": generated_text,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        usage=usage,
    )
    return JSONResponse(content=jsonable_encoder(completion), headers={"x-cms-tracking-id": tracking_id})


@router.post(
    PATH_EMBEDDINGS,
    tags=[Tags.OpenAICompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Create embeddings based on text(s), similar to OpenAI's /v1/embeddings endpoint",
)
def embed_texts(
    request: Request,
    request_data: Annotated[OpenAIEmbeddingsRequest, Body(
        description="Text(s) to be embedded", media_type="application/json"
    )],
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> JSONResponse:
    """
    Embeds text or a list of texts, mimicking OpenAI's /v1/embeddings endpoint.

    Args:
        request (Request): The request object.
        request_data (OpenAIEmbeddingsRequest): The request data containing model and input text(s).
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A response containing the embeddings of the text(s).
    """
    tracking_id = tracking_id or str(uuid.uuid4())

    if not hasattr(model_service, "create_embeddings"):
        error_response = {
            "error": {
                "message": "Model does not support embeddings",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_supported",
            }
        }
        return JSONResponse(
            content=error_response,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"x-cms-tracking-id": tracking_id},
        )

    input_text = request_data.input
    model = model_service.model_name if request_data.model != model_service.model_name else request_data.model

    if isinstance(input_text, str):
        input_texts = [input_text]
    else:
        input_texts = input_text

    try:
        embeddings_data = []

        for i, embedding in enumerate(model_service.create_embeddings(input_texts)):
            embeddings_data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i,
            })

        response = OpenAIEmbeddingsResponse(object="list", data=embeddings_data, model=model)

        return JSONResponse(
            content=jsonable_encoder(response),
            headers={"x-cms-tracking-id": tracking_id},
        )

    except Exception as e:
        logger.error("Failed to create embeddings")
        logger.exception(e)
        error_response = {
            "error": {
                "message": f"Failed to create embeddings: {str(e)}",
                "type": "server_error",
                "code": "internal_error",
            }
        }
        return JSONResponse(
            content=error_response,
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"x-cms-tracking-id": tracking_id},
        )


def _empty_prompt_error() -> Iterable[str]:
    yield "ERROR: No prompt text provided\n"


def _send_usage_metrics(handler: str, prompt_token_num: int, completion_token_num: int) -> None:
    cms_prompt_tokens.labels(handler=handler).observe(prompt_token_num)
    logger.debug("Sent prompt tokens usage: %s", prompt_token_num)
    cms_completion_tokens.labels(handler=handler).observe(completion_token_num)
    logger.debug("Sent completion tokens usage: %s", completion_token_num)
    cms_total_tokens.labels(handler=handler).observe(prompt_token_num + completion_token_num)
    logger.debug("Sent total tokens usage: %s", prompt_token_num + completion_token_num)
