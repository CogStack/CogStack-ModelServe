import json
import logging
import time
import uuid
import app.api.globals as cms_globals

from typing import Union, Iterable, AsyncGenerator, List, Optional, Dict, Any, cast
from typing_extensions import Annotated
from functools import partial
from fastapi import APIRouter, Depends, Request, Body, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse, Response
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_404_NOT_FOUND,
)
from app.domain import (
    Tags,
    TagsGenerative,
    GenerationResult,
    OpenAIChatCompletionsRequest,
    OpenAIChatCompletionsResponse,
    OpenAICompletionsRequest,
    OpenAICompletionsResponse,
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
    OpenAIFunctionTool,
    OpenAIResponseFormat,
    PromptMessage,
    PromptRole,
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
    OllamaEmbedRequest,
)
from app.model_services.base import AbstractModelService
from app.utils import (
    get_settings,
    get_prompt_from_messages,
    get_default_chat_template,
    resolve_safe_max_model_length,
    utilise_local_chat_template,
    dump_pydantic_object_to_dict,
    extract_tool_calls,
)
from app.api.utils import get_rate_limiter
from app.api.dependencies import validate_tracking_id
from app.management.prometheus_metrics import (
    cms_prompt_tokens,
    cms_completion_tokens,
    cms_total_tokens,
    cms_ttft_milliseconds,
    cms_tpot_milliseconds,
)
from app.exception import GenerationException, ClientException


PATH_GENERATE = "/generate"
PATH_GENERATE_ASYNC = "/stream/generate"

# OpenAI-compatible endpoints
PATH_CHAT_COMPLETIONS = "/openai/v1/chat/completions"
PATH_COMPLETIONS = "/openai/v1/completions"
PATH_EMBEDDINGS = "/openai/v1/embeddings"
PATH_MODELS = "/openai/v1/models"

# Ollama-compatible endpoints
PATH_OLLAMA_ROOT = "/ollama/"
PATH_OLLAMA_TAGS = "/ollama/api/tags"
PATH_OLLAMA_CHAT = "/ollama/api/chat"
PATH_OLLAMA_GENERATE = "/ollama/api/generate"
PATH_OLLAMA_SHOW = "/ollama/api/show"
PATH_OLLAMA_VERSION = "/ollama/api/version"
PATH_OLLAMA_EMBED = "/ollama/api/embed"

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
@limiter.limit(config.GENERATION_RATE_LIMIT)
def generate_text(
    request: Request,
    prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
    max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512,
    temperature: Annotated[float, Query(description="The temperature of the generated text", ge=0.0)] = 0.7,
    top_p: Annotated[float, Query(description="The Top-P value for nucleus sampling", ge=0.0, le=1.0)] = 0.9,
    stop_sequences: Annotated[List[str], Query(description="The list of sequences used to stop the generation")] = [],
    include_usage: Annotated[bool, Query(description="Whether to include token usage in the response")] = False,
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    chat_template: Annotated[Optional[str], Query(description="Override chat template for prompt formatting")] = None,
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
        include_usage (bool): Whether to include token usage in the response.
        ensure_full_sentences (bool): Whether to generate full sentences only.
        chat_template  (Optional[str]): Override chat template name for prompt formatting.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        PlainTextResponse: A response containing the generated text.
    """

    tracking_id = tracking_id or str(uuid.uuid4())

    if prompt:
        generation_result: GenerationResult = model_service.generate(
            _build_prompt_text(
                model_service,
                prompt,
                override_template=chat_template if chat_template else None,
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            report_tokens=partial(_send_usage_metrics, handler=PATH_GENERATE),
            ensure_full_sentences=ensure_full_sentences,
        )

        return PlainTextResponse(
            generation_result.text,
            headers={
                "x-cms-tracking-id": tracking_id,
                "x-cms-gen-prompt-token-num": str(generation_result.prompt_token_num),
                "x-cms-gen-completion-token-num": str(generation_result.completion_token_num),
                "x-cms-gen-total-token-num": (
                    str(generation_result.prompt_token_num + generation_result.completion_token_num)
                ),
                "x-cms-gen-tpot-ms": str(generation_result.tpot_ms),
            } if include_usage else {"x-cms-tracking-id": tracking_id},
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
@limiter.limit(config.GENERATION_RATE_LIMIT)
async def generate_text_stream(
    request: Request,
    prompt: Annotated[str, Body(description="The prompt to be sent to the model", media_type="text/plain")],
    max_tokens: Annotated[int, Query(description="The maximum number of tokens to generate", gt=0)] = 512,
    temperature: Annotated[float, Query(description="The temperature of the generated text", ge=0.0)] = 0.7,
    top_p: Annotated[float, Query(description="The Top-P value for nucleus sampling", ge=0.0, le=1.0)] = 0.9,
    stop_sequences: Annotated[List[str], Query(description="The list of sequences used to stop the generation")] = [],
    include_usage: Annotated[bool, Query(description="Whether to include token usage in the response")] = False,
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    chat_template: Annotated[Optional[str], Query(description="Override chat template for prompt formatting")] = None,
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
        include_usage (bool): Whether to include token usage in the response.
        ensure_full_sentences (bool): Whether to generate full sentences only.
        chat_template (Optional[str]): Override chat template for prompt formatting.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A streaming response containing the text generated in near real-time.
    """

    tracking_id = tracking_id or str(uuid.uuid4())
    prompt = _build_prompt_text(
        model_service,
        prompt,
        override_template=chat_template if chat_template else None,
    )

    async def _stream(prompt: str) -> AsyncGenerator:
        async for generated in model_service.generate_async(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            report_tokens=partial(_send_usage_metrics, handler=PATH_GENERATE_ASYNC),
            ensure_full_sentences=ensure_full_sentences,
        ):
            if isinstance(generated, GenerationResult):
                if include_usage:
                    yield (
                       "\n\n<cms_token_usage>"
                       f"Prompt tokens: {str(generated.prompt_token_num)}; "
                       f"Completion tokens: {str(generated.completion_token_num)}; "
                       f"Total tokens: {str(generated.prompt_token_num + generated.completion_token_num)}; "
                       f"TTFT in milliseconds: {str(generated.ttft_ms)}; "
                       f"TPOT in milliseconds: {str(generated.tpot_ms)}"
                       "</cms_token_usage>"
                    )
                continue
            yield generated

    if prompt:
        return StreamingResponse(
            _stream(prompt),
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
@limiter.limit(config.GENERATION_RATE_LIMIT)
def generate_chat_completions(
    request: Request,
    request_data: Annotated[OpenAIChatCompletionsRequest, Body(
        description="OpenAI-like completion request", media_type="application/json"
    )],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    chat_template: Annotated[Optional[str], Query(description="Override chat template for prompt formatting")] = None,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Generates chat response based on messages, mimicking OpenAI's /v1/chat/completions endpoint.

    Args:
        request (Request): The request object.
        request_data (OpenAIChatRequest): The request data containing model, messages, stream, temperature, top_p, and stop_sequences.
        ensure_full_sentences (bool): Whether to generate full sentences only.
        chat_template (Optional[str]): Override chat template for prompt formatting.
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: A OpenAI-like response containing the text generated in near real-time.
        JSONResponse: A response containing an error message if the prompt messages are empty.
    """

    messages = request_data.messages
    tools = [dump_pydantic_object_to_dict(tool) for tool in request_data.tools] if request_data.tools else None
    tools_for_prompt = cast(Optional[List[Union[OpenAIFunctionTool, Dict[Any, Any]]]], tools)
    model = model_service.model_name if request_data.model != model_service.model_name else request_data.model
    stream = request_data.stream
    include_usage = request_data.stream_options.include_usage if request_data.stream_options else False
    max_tokens = request_data.max_tokens
    temperature = request_data.temperature
    top_p = request_data.top_p
    json_schema_parser = _get_parser_for_response_format(request_data.response_format)

    if isinstance(request_data.stop, str):
        stop_sequences = [request_data.stop]
    elif isinstance(request_data.stop, list):
        stop_sequences = request_data.stop
    else:
        stop_sequences = []
    tracking_id = tracking_id or str(uuid.uuid4())
    _ensures_chat_template(model_service, chat_template)

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
        prefix_prompt: Optional[str],
    ) -> AsyncGenerator:
        data: Dict[str, Any] = {
            "id": tracking_id,
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"role": PromptRole.ASSISTANT.value}}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        stream_buffer = ""
        tool_call_emitted = False
        try:
            async for generated in model_service.generate_async(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                report_tokens=partial(_send_usage_metrics, handler=PATH_CHAT_COMPLETIONS),
                ensure_full_sentences=ensure_full_sentences,
                json_schema_parser=json_schema_parser,
                prefix_prompt=prefix_prompt,
            ):
                if isinstance(generated, GenerationResult):
                    if include_usage:
                        data = {
                            "usage": {
                                "prompt_tokens": generated.prompt_token_num,
                                "completion_tokens": generated.completion_token_num,
                                "total_tokens": generated.prompt_token_num + generated.completion_token_num,
                            }
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    continue
                if tool_call_emitted:
                    continue
                if tools_for_prompt:
                    stream_buffer += generated
                    tool_calls = extract_tool_calls(stream_buffer)
                    if tool_calls:
                        data = {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "id": tool_call["id"],
                                                "type": "function",
                                                "function": tool_call["function"],
                                            }
                                            for tool_call in tool_calls
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                            "object": "chat.completion.chunk",
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        tool_call_emitted = True
                        continue
                data = {
                    "choices": [
                        {
                            "delta": {"content": generated}
                        }
                    ],
                    "object": "chat.completion.chunk",
                }
                yield f"data: {json.dumps(data)}\n\n"
        except GenerationException as e:
            logger.error("Streaming chat generation failed for tracking_id=%s", tracking_id, exc_info=e)
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "generation_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

    assert hasattr(model_service, "tokenizer"), "Model service doesn't have a tokenizer"
    prompt = get_prompt_from_messages(
        tokenizer=model_service.tokenizer,
        messages=messages,
        tools=tools_for_prompt,
        max_input_tokens=(resolve_safe_max_model_length(model_service.model.config) - max_tokens),  # type: ignore
    )
    prefix_prompt = None
    if messages and messages[0].role == PromptRole.SYSTEM:
        prefix_prompt = get_prompt_from_messages(
            tokenizer=model_service.tokenizer,
            messages=[messages[0]],
            tools=tools_for_prompt,
            add_generation_prompt=False,
        )
    if stream:
        return StreamingResponse(
            _stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences or [],
                ensure_full_sentences=ensure_full_sentences,
                prefix_prompt=prefix_prompt,
            ),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
        )
    else:
        def _report_tokens(
            prompt_token_num: int,
            completion_token_num: int,
            ttft_milliseconds: int = -1,
            tpot_milliseconds: int = -1,
        ) -> None:
            _send_usage_metrics(
                handler=PATH_CHAT_COMPLETIONS,
                prompt_token_num=prompt_token_num,
                completion_token_num=completion_token_num,
                ttft_milliseconds=ttft_milliseconds,
                tpot_milliseconds=tpot_milliseconds,
            )
        generation_result = model_service.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences or [],
            report_tokens=_report_tokens,
            ensure_full_sentences=ensure_full_sentences,
            json_schema_parser=json_schema_parser,
            prefix_prompt=prefix_prompt,
        )
        tool_calls = extract_tool_calls(generation_result.text) if tools_for_prompt else []
        if tool_calls:
            choices = [
                {
                    "index": 0,
                    "message": {
                        "role": PromptRole.ASSISTANT.value,
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        else:
            choices = [
                {
                    "index": 0,
                    "message": PromptMessage(
                        role=PromptRole.ASSISTANT,
                        content=generation_result.text,
                    ),
                    "finish_reason": "stop",
                }
            ]

        completion = OpenAIChatCompletionsResponse(
            id=tracking_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage={
                "prompt_tokens": generation_result.prompt_token_num,
                "completion_tokens": generation_result.completion_token_num,
                "total_tokens": generation_result.prompt_token_num + generation_result.completion_token_num,
            } if include_usage else None,
        )
        return JSONResponse(content=jsonable_encoder(completion), headers={"x-cms-tracking-id": tracking_id})


@router.post(
    PATH_COMPLETIONS,
    tags=[Tags.OpenAICompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate completion based on prompt, similar to OpenAI's /v1/completions",
)
@limiter.limit(config.GENERATION_RATE_LIMIT)
def generate_text_completions(
    request: Request,
    request_data: Annotated[OpenAICompletionsRequest, Body(
        description="OpenAI-like completion request", media_type="application/json"
    )],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    chat_template: Annotated[Optional[str], Query(description="Override chat template for prompt formatting")] = None,
    tracking_id: Union[str, None] = Depends(validate_tracking_id),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Generates completion response based on prompt, mimicking OpenAI's /v1/completions endpoint.

    Args:
        request (Request): The request object.
        request_data (OpenAICompletionsRequest): The request data containing model, prompt, stream, temperature, top_p, and stop.
        ensure_full_sentences (bool): Whether to generate full sentences only.
        chat_template (Optional[str]): Override chat template for prompt formatting
        tracking_id (Union[str, None]): An optional tracking ID of the requested task.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        StreamingResponse: An OpenAI-like streaming response.
        JSONResponse: A response containing the generated text or an error message.
    """

    tracking_id = tracking_id or str(uuid.uuid4())
    model = model_service.model_name if request_data.model != model_service.model_name else request_data.model
    stream = request_data.stream
    include_usage = request_data.stream_options.include_usage if request_data.stream_options else False
    max_tokens = request_data.max_tokens
    temperature = request_data.temperature
    top_p = request_data.top_p

    if isinstance(request_data.stop, str):
        stop_sequences = [request_data.stop]
    elif isinstance(request_data.stop, list):
        stop_sequences = request_data.stop
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
        data: Dict[str, Any] = {
            "id": tracking_id,
            "object": "text_completion",
            "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        try:
            async for generated in model_service.generate_async(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                report_tokens=partial(_send_usage_metrics, handler=PATH_COMPLETIONS),
                ensure_full_sentences=ensure_full_sentences,
            ):
                if isinstance(generated, GenerationResult):
                    if include_usage:
                        data = {
                            "usage": {
                                "prompt_tokens": generated.prompt_token_num,
                                "completion_tokens": generated.completion_token_num,
                                "total_tokens": generated.prompt_token_num + generated.completion_token_num,
                            }
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    continue
                data = {
                    "object": "text_completion",
                    "choices": [
                        {
                            "text": generated,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
        except GenerationException as e:
            logger.error("Streaming completion generation failed for tracking_id=%s", tracking_id, exc_info=e)
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "generation_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

    prompt = _build_prompt_text(
        model_service,
        prompt,
        override_template=chat_template if chat_template else None,
    )
    if stream:
        return StreamingResponse(
            _stream(prompt, max_tokens, temperature, top_p, stop_sequences, ensure_full_sentences),
            media_type="text/event-stream",
            headers={"x-cms-tracking-id": tracking_id},
        )
    else:
        def _report_tokens(
            prompt_token_num: int,
            completion_token_num: int,
            ttft_milliseconds: int = -1,
            tpot_milliseconds: int = -1,
        ) -> None:
            _send_usage_metrics(
                handler=PATH_COMPLETIONS,
                prompt_token_num=prompt_token_num,
                completion_token_num=completion_token_num,
                ttft_milliseconds=ttft_milliseconds,
                tpot_milliseconds=tpot_milliseconds,
            )

        generation_result = model_service.generate(
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
                    "text": generation_result.text,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": generation_result.prompt_token_num,
                "completion_tokens": generation_result.completion_token_num,
                "total_tokens": generation_result.prompt_token_num + generation_result.completion_token_num,
            } if include_usage else None,
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


@router.get(
    PATH_MODELS,
    tags=[Tags.OpenAICompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="List available models, similar to OpenAI's /v1/models endpoint",
)
async def list_models(
    request: Request,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> JSONResponse:
    """
    Lists all available models, mimicking OpenAI's /v1/models endpoint.

    Args:
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A response containing the list of models.
    """
    response = {
        "object": "list",
        "data": [
            {
                "id": model_service.model_name.replace(" ", "_"),
                "object": "model",
                "created": 0,
                "owned_by": "cms",
            }
        ],
    }
    return JSONResponse(content=response)


@router.get(
    PATH_MODELS + "/{model_name}",
    tags=[Tags.OpenAICompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Get a specific model, similar to OpenAI's /v1/models/{model_id} endpoint",
)
async def get_model(
    request: Request,
    model_name: str,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)
) -> JSONResponse:
    """
    Gets a specific model by ID, mimicking OpenAI's /v1/models/{model_id} endpoint.

    Args:
        model_name (str): The model name to retrieve.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A response containing the model details.
    """
    if model_name != model_service.model_name.replace(" ", "_"):
        error_response = {
            "error": {
                "message": f"The model `{model_name}` does not exist",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        }
        return JSONResponse(content=error_response, status_code=HTTP_404_NOT_FOUND
)
    response = {
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "cms",
        "permission": [],
        "root": model_name,
        "parent": None,
    }
    return JSONResponse(content=response)


@router.get(
    PATH_OLLAMA_ROOT,
    tags=[Tags.OllamaCompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Health check, similar to Ollama's / endpoint (GET)",
)
async def ollama_health_get(request: Request) -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=HTTP_200_OK)


@router.head(
    PATH_OLLAMA_ROOT,
    tags=[Tags.OllamaCompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Health check, similar to Ollama's / endpoint (HEAD)",
)
async def ollama_health_head(request: Request) -> Response:
    return Response(status_code=HTTP_200_OK)


@router.get(
    PATH_OLLAMA_VERSION,
    tags=[Tags.OllamaCompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Get the API version, similar to Ollama's /api/version endpoint",
)
async def ollama_version(
    request: Request,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    return JSONResponse(content={"version": model_service.api_version}) # type: ignore


@router.get(
    PATH_OLLAMA_TAGS,
    tags=[Tags.OllamaCompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="List available models, similar to Ollama's /api/tags endpoint",
)
async def ollama_list_tags(
    request: Request,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    model_name = model_service.model_name.replace(" ", "_")
    response = {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "digest": model_service.digest, # type: ignore
                "details": {
                    "format": "cmsmp",
                    "family": model_service.model.config.model_type,    # type: ignore
                    "families": [model_service.model.config.model_type],    # type: ignore
                },
            }
        ]
    }
    return JSONResponse(content=response)


@router.post(
    PATH_OLLAMA_SHOW,
    tags=[Tags.OllamaCompatible],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Show model information, similar to Ollama's /api/show endpoint",
)
async def ollama_show_model(
    request: Request,
    request_data: Annotated[OllamaShowRequest, Body(description="Ollama show request", media_type="application/json")],
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    requested_model_name = request_data.model
    model_name = model_service.model_name.replace(" ", "_")
    if requested_model_name and requested_model_name != model_name:
        return JSONResponse(
            content={"error": f"model '{requested_model_name}' not found"},
            status_code=HTTP_404_NOT_FOUND,
        )

    model_card = model_service.info().model_card

    response = {
        "modelfile": model_service.info().model_card.get(   # type: ignore
            "_name_or_path", model_service.model_name
        ),
        "template": model_service.tokenizer.chat_template,  # type: ignore
        "details": {
            "family": model_service.model.config.model_type,    # type: ignore
        },
        "model_info": model_card,
        "capabilities": ["completion", "chat", "create_embeddings"]
    }
    return JSONResponse(content=response)


@router.post(
    PATH_OLLAMA_GENERATE,
    tags=[Tags.OllamaCompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate a completion, similar to Ollama's /api/generate endpoint",
)
@limiter.limit(config.GENERATION_RATE_LIMIT)
async def ollama_generate(
    request: Request,
    request_data: Annotated[
        OllamaGenerateRequest, Body(description="Ollama generate request", media_type="application/json")
    ],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    chat_template: Annotated[Optional[str], Query(description="Override chat template for prompt formatting")] = None,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> Union[StreamingResponse, JSONResponse]:
    prompt = request_data.prompt
    if not prompt:
        return JSONResponse(content={"error": "prompt is required"}, status_code=HTTP_400_BAD_REQUEST)

    stream = request_data.stream
    model_name = model_service.model_name.replace(" ", "_")
    options = request_data.options.model_dump(exclude_none=True) if request_data.options else {}
    max_tokens = options.get("num_predict", 512)
    temperature = options.get("temperature", 0.7)
    top_p = options.get("top_p", 0.9)
    stop_sequences = _normalise_stop_sequences(options.get("stop", None))
    json_schema_parser = _get_parser_for_json_schema(request_data.format)

    def _report_tokens(
        prompt_token_num: int,
        completion_token_num: int,
        ttft_milliseconds: int = -1,
        tpot_milliseconds: int = -1,
    ) -> None:
        _send_usage_metrics(
            PATH_OLLAMA_GENERATE,
            prompt_token_num,
            completion_token_num,
            ttft_milliseconds=ttft_milliseconds,
            tpot_milliseconds=tpot_milliseconds,
        )

    prompt = _build_prompt_text(
        model_service,
        prompt,
        override_template=chat_template if chat_template else None,
    )
    if stream:
        async def _stream() -> AsyncGenerator[str, None]:
            start = time.perf_counter_ns()
            generation_result: Optional[GenerationResult] = None
            try:
                async for generated in model_service.generate_async(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    report_tokens=_report_tokens,
                    ensure_full_sentences=ensure_full_sentences,
                    json_schema_parser=json_schema_parser,
                ):
                    if isinstance(generated, GenerationResult):
                        generation_result = generated
                    else:
                        yield json.dumps({
                            "model": model_name,
                            "created_at": _iso_utc_now(),
                            "response": generated,
                            "done": False,
                        }) + "\n"
            except GenerationException as e:
                logger.error("Ollama stream generation failed", exc_info=e)
                yield json.dumps({
                    "model": model_name,
                    "created_at": _iso_utc_now(),
                    "response": "",
                    "done": True,
                    "done_reason": "error",
                    "error": str(e),
                    "total_duration": time.perf_counter_ns() - start,
                }) + "\n"
                return
            yield json.dumps({
                "model": model_name,
                "created_at": _iso_utc_now(),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": generation_result.prompt_token_num if generation_result is not None else 0,
                "eval_count": generation_result.completion_token_num if generation_result is not None else 0,
                "ttft_in_milliseconds": generation_result.ttft_ms if generation_result is not None else -1,
                "tpot_in_milliseconds": generation_result.tpot_ms if generation_result is not None else -1,
                "total_duration": time.perf_counter_ns() - start,
            }) + "\n"

        return StreamingResponse(_stream(), media_type="application/x-ndjson")
    else:
        start = time.perf_counter_ns()
        generation_result = model_service.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            report_tokens=_report_tokens,
            ensure_full_sentences=ensure_full_sentences,
            json_schema_parser=json_schema_parser,
        )
        return JSONResponse(content={
            "model": model_name,
            "created_at": _iso_utc_now(),
            "response": generation_result.text,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": generation_result.prompt_token_num,
            "eval_count": generation_result.completion_token_num,
            "ttft_in_milliseconds": generation_result.ttft_ms,
            "tpot_in_milliseconds": generation_result.tpot_ms,
            "total_duration": time.perf_counter_ns() - start,
        })


@router.post(
    PATH_OLLAMA_CHAT,
    tags=[Tags.OllamaCompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Generate a chat completion, similar to Ollama's /api/chat endpoint",
)
@limiter.limit(config.GENERATION_RATE_LIMIT)
async def ollama_chat(
    request: Request,
    request_data: Annotated[OllamaChatRequest, Body(description="Ollama chat request", media_type="application/json")],
    ensure_full_sentences: Annotated[bool, Query(description="Whether to generate full sentences only")] = False,
    chat_template: Annotated[Optional[str], Query(description="Override chat template for prompt formatting")] = None,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> Union[StreamingResponse, JSONResponse]:
    raw_messages = request_data.messages
    if not raw_messages:
        return JSONResponse(content={"error": "messages are required"}, status_code=HTTP_400_BAD_REQUEST)

    stream = request_data.stream
    model_name = model_service.model_name.replace(" ", "_")
    options = request_data.options.model_dump(exclude_none=True) if request_data.options else {}
    max_tokens = options.get("num_predict", 512)
    temperature = options.get("temperature", 0.7)
    top_p = options.get("top_p", 0.9)
    stop_sequences = _normalise_stop_sequences(options.get("stop", None))
    json_schema_parser = _get_parser_for_json_schema(request_data.format)
    _ensures_chat_template(model_service, chat_template)

    def _report_tokens(
        prompt_token_num: int,
        completion_token_num: int,
        ttft_milliseconds: int = -1,
        tpot_milliseconds: int = -1,
    ) -> None:
        _send_usage_metrics(
            PATH_OLLAMA_CHAT,
            prompt_token_num,
            completion_token_num,
            ttft_milliseconds=ttft_milliseconds,
            tpot_milliseconds=tpot_milliseconds,
        )

    prompt_messages: List[PromptMessage] = []
    for message in raw_messages:
        role_text = message.role
        try:
            role = PromptRole(role_text)
        except ValueError:
            role = PromptRole.USER
        prompt_messages.append(PromptMessage(role=role, content=str(message.content)))
    prompt = get_prompt_from_messages(
        tokenizer=model_service.tokenizer,  # type: ignore
        messages=prompt_messages,
        max_input_tokens=(resolve_safe_max_model_length(model_service.model.config) - max_tokens),  # type: ignore
    )
    prefix_prompt = None
    if prompt_messages and prompt_messages[0].role == PromptRole.SYSTEM:
        prefix_prompt = get_prompt_from_messages(
            tokenizer=model_service.tokenizer,  # type: ignore
            messages=[prompt_messages[0]],
            add_generation_prompt=False,
        )

    if stream:
        start = time.perf_counter_ns()

        async def _stream() -> AsyncGenerator[str, None]:
            generated_result: Optional[GenerationResult] = None
            try:
                async for generated in model_service.generate_async(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    report_tokens=_report_tokens,
                    ensure_full_sentences=ensure_full_sentences,
                    json_schema_parser=json_schema_parser,
                    prefix_prompt=prefix_prompt,
                ):
                    if isinstance(generated, GenerationResult):
                        generated_result = generated
                    else:
                        yield json.dumps({
                            "model": model_name,
                            "created_at": _iso_utc_now(),
                            "message": {"role": PromptRole.ASSISTANT.value, "content": generated},
                            "done": False,
                        }) + "\n"
            except GenerationException as e:
                logger.error("Ollama chat stream generation failed", exc_info=e)
                yield json.dumps({
                    "model": model_name,
                    "created_at": _iso_utc_now(),
                    "message": {"role": PromptRole.ASSISTANT.value, "content": ""},
                    "done": True,
                    "done_reason": "error",
                    "error": str(e),
                    "total_duration": time.perf_counter_ns() - start,
                }) + "\n"
                return
            yield json.dumps({
                "model": model_name,
                "created_at": _iso_utc_now(),
                "message": {"role": PromptRole.ASSISTANT.value, "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": generated_result.prompt_token_num if generated_result is not None else 0,
                "eval_count": generated_result.completion_token_num if generated_result is not None else 0,
                "ttft_in_milliseconds": generated_result.ttft_ms if generated_result is not None else -1,
                "tpot_in_milliseconds": generated_result.tpot_ms if generated_result is not None else -1,
                "total_duration": time.perf_counter_ns() - start,
            }) + "\n"

        return StreamingResponse(_stream(), media_type="application/x-ndjson")
    else:
        start = time.perf_counter_ns()
        generated_result = model_service.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            report_tokens=_report_tokens,
            ensure_full_sentences=ensure_full_sentences,
            json_schema_parser=json_schema_parser,
            prefix_prompt=prefix_prompt,
        )

        return JSONResponse(content={
            "model": model_name,
            "created_at": _iso_utc_now(),
            "message": {"role": PromptRole.ASSISTANT.value, "content": generated_result.text},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": generated_result.prompt_token_num,
            "eval_count": generated_result.completion_token_num,
            "ttft_in_milliseconds": generated_result.ttft_ms,
            "tpot_in_milliseconds": generated_result.tpot_ms,
            "total_duration": time.perf_counter_ns() - start,
        })


@router.post(
    PATH_OLLAMA_EMBED,
    tags=[Tags.OllamaCompatible],
    response_model=None,
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Create embeddings, similar to Ollama's /api/embed endpoint",
)
def ollama_embed(
    request: Request,
    request_data: Annotated[OllamaEmbedRequest, Body(description="Ollama embed request", media_type="application/json")],
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    inputs = [request_data.input] if isinstance(request_data.input, str) else request_data.input
    start = time.perf_counter_ns()
    embeddings = model_service.create_embeddings(inputs)
    return JSONResponse(content={
        "model": model_service.model_name.replace(" ", "_"),
        "embeddings": embeddings,
        "total_duration": time.perf_counter_ns() - start,
    })


def _empty_prompt_error() -> Iterable[str]:
    yield "ERROR: No prompt text provided\n"


def _ensures_chat_template(
    model_service: AbstractModelService,
    override_template: Optional[str],
) -> None:
    assert hasattr(model_service, "tokenizer"), "Model service doesn't have a tokenizer"
    if override_template:
        model_service.tokenizer.chat_template = override_template  # type: ignore
        return
    if hasattr(model_service.tokenizer, "chat_template") and model_service.tokenizer.chat_template is None:  # type: ignore
        model_type = model_service.model.config.model_type  # type: ignore
        used_local_template = utilise_local_chat_template(model_type, model_service.tokenizer)  # type: ignore
        if not used_local_template:
            model_service.tokenizer.chat_template = get_default_chat_template()  # type: ignore


def _build_prompt_text(
    model_service: AbstractModelService,
    prompt: str,
    override_template: Optional[str] = None,
) -> str:
    _ensures_chat_template(model_service, override_template)
    return get_prompt_from_messages(
        tokenizer=model_service.tokenizer,  # type: ignore
        messages=[PromptMessage(role=PromptRole.USER, content=prompt)],
        add_generation_prompt=True,
    )


def _get_parser_for_response_format(response_format: Optional[OpenAIResponseFormat]) -> Optional[Any]:
    if response_format is None:
        return None
    if response_format.type == "json_schema":
        try:
            from lmformatenforcer import JsonSchemaParser
            parser = JsonSchemaParser(response_format.json_schema.schema_)
            setattr(parser, "schema", response_format.json_schema.schema_)
            return parser
        except ImportError as e:
            raise ClientException("lmformatenforcer package is not installed; required for JSON schema support") from e
        except Exception as exc:
            raise ClientException("Invalid JSON schema in response_format") from exc
    else:
        raise ClientException("Unsupported response_format type; only 'json_schema' is supported")


def _get_parser_for_json_schema(json_schema: Optional[Dict[str, Any]]) -> Optional[Any]:
    if json_schema is None:
        return None
    try:
        from lmformatenforcer import JsonSchemaParser
        return JsonSchemaParser(json_schema)
    except ImportError as e:
        raise ClientException("lmformatenforcer package is not installed; required for JSON schema support") from e
    except Exception as exc:
        raise ClientException("Invalid JSON schema") from exc


def _send_usage_metrics(
    handler: str,
    prompt_token_num: int,
    completion_token_num: int,
    ttft_milliseconds: int = -1,
    tpot_milliseconds: int = -1,
) -> None:
    cms_prompt_tokens.labels(handler=handler).observe(prompt_token_num)
    logger.debug("Sent prompt tokens usage: %s", prompt_token_num)
    cms_completion_tokens.labels(handler=handler).observe(completion_token_num)
    logger.debug("Sent completion tokens usage: %s", completion_token_num)
    cms_total_tokens.labels(handler=handler).observe(prompt_token_num + completion_token_num)
    logger.debug("Sent total tokens usage: %s", prompt_token_num + completion_token_num)
    if ttft_milliseconds != -1:
        cms_ttft_milliseconds.labels(handler=handler).observe(ttft_milliseconds)
        logger.debug("Sent time to first token: %s ms", ttft_milliseconds)
    if tpot_milliseconds != -1:
        cms_tpot_milliseconds.labels(handler=handler).observe(tpot_milliseconds)
        logger.debug("Sent time per output token: %s ms", tpot_milliseconds)


def _iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def _normalise_stop_sequences(raw_stop: Union[None, str, List[str]]) -> List[str]:
    if isinstance(raw_stop, str):
        return [raw_stop]
    if isinstance(raw_stop, list):
        return [str(x) for x in raw_stop]
    return []
