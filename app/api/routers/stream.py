import json
import logging
import asyncio
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.websockets import WebSocketDisconnect
from starlette.requests import ClientDisconnect

import app.api.globals as cms_globals

from typing import Any, Mapping, Optional, AsyncGenerator
from starlette.types import Receive, Scope, Send
from starlette.background import BackgroundTask
from fastapi import APIRouter, Depends, Request, Response, WebSocket, WebSocketException
from pydantic import ValidationError
from app.domain import Tags, TextStreamItem
from app.model_services.base import AbstractModelService
from app.utils import get_settings
from app.api.utils import get_rate_limiter
from app.api.auth.users import get_user_manager, CmsUserManager

PATH_STREAM_PROCESS = "/process"
PATH_WS = "/ws"
PATH_GENERATE= "/generate"

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)
logger = logging.getLogger("cms")

assert cms_globals.props is not None, "Current active user dependency not injected"
assert cms_globals.model_service_dep is not None, "Model service dependency not injected"


@router.post(
    PATH_STREAM_PROCESS,
    tags=[Tags.Annotations.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="Extract the NER entities from a stream of texts in the JSON Lines format",
)
@limiter.limit(config.PROCESS_BULK_RATE_LIMIT)
async def get_entities_stream_from_jsonlines_stream(
    request: Request,
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> Response:
    """
    Extracts NER entities from a stream of texts in the JSON Lines format and returns them as a JSON Lines stream.

    Args:
        request (Request): The request object.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        Response: A streaming response containing the original texts and extracted entities in the JSON Lines format.
    """

    annotation_stream = _annotation_async_gen(request, model_service)
    return _LocalStreamingResponse(annotation_stream, media_type="application/x-ndjson; charset=utf-8")


@router.websocket(PATH_WS)
# @limiter.limit(config.PROCESS_BULK_RATE_LIMIT)  # Not supported yet
async def get_inline_annotations_from_websocket(
    websocket: WebSocket,
    user_manager: CmsUserManager = Depends(get_user_manager),
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> None:
    """
    Handles WebSocket connections for receiving text and returning extracted NER entities.

    This endpoint establishes a WebSocket connection to receive text data from the client,
    processes the text to extract NER entities using the provided model service, and sends
    the extracted entities back to the client. The connection will be closed if no messages are
    received within the specified idle timeout duration.

    Args:
        websocket (WebSocket): The WebSocket connection object.
        user_manager (CmsUserManager): The user manager dependency for handling user authentication.
        model_service (AbstractModelService): The model service dependency.

    Raises:
        WebSocketException: If the authentication cookie is not found or the user is not active.
    """

    monitor_idle_task = None
    try:
        if get_settings().AUTH_USER_ENABLED == "true":
            cookie = websocket.cookies.get("fastapiusersauth")
            if cookie is None:
                raise WebSocketException(code=WS_1008_POLICY_VIOLATION, reason="Authentication cookie not found")
            user = await cms_globals.props.auth_backends[1].get_strategy().read_token(cookie, user_manager) # type: ignore
            if not user or not user.is_active:
                raise WebSocketException(code=WS_1008_POLICY_VIOLATION, reason="User not found or not active")

        await websocket.accept()

        time_of_last_seen_msg = asyncio.get_event_loop().time()

        async def _monitor_idle() -> None:
            while True:
                await asyncio.sleep(get_settings().WS_IDLE_TIMEOUT_SECONDS)
                if (asyncio.get_event_loop().time() - time_of_last_seen_msg) >= get_settings().WS_IDLE_TIMEOUT_SECONDS:
                    await websocket.close()
                    logger.debug("Connection closed due to inactivity")
                    break

        monitor_idle_task = asyncio.create_task(_monitor_idle())

        while True:
            text = await websocket.receive_text()
            time_of_last_seen_msg = asyncio.get_event_loop().time()
            try:
                annotations = await model_service.annotate_async(text)
                annotated_text = ""
                start_index = 0
                for anno in annotations:
                    annotated_text += f'{text[start_index:anno.start]}[{anno.label_name}: {text[anno.start:anno.end]}]'
                    start_index = anno.end
                annotated_text += text[start_index:]
            except Exception as e:
                await websocket.send_text(f"ERROR: {str(e)}")
            else:
                await websocket.send_text(annotated_text)
    except WebSocketDisconnect as e:
        logger.debug(str(e))
    finally:
        try:
            if monitor_idle_task is not None:
                monitor_idle_task.cancel()
            await websocket.close()
        except RuntimeError as e:
            logger.debug(str(e))


class _LocalStreamingResponse(Response):

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        max_chunk_size: int = 1024,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.max_chunk_size = max_chunk_size
        if media_type is not None:
            self.media_type = media_type
        self.background = background
        self.init_headers(headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        response_started = False
        async for line in self.content:
            if not response_started:
                await send({
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                })
                response_started = True
            line_bytes = line.encode("utf-8")
            for i in range(0, len(line_bytes), self.max_chunk_size):
                chunk = line_bytes[i:i + self.max_chunk_size]
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                })
        if not response_started:
            await send({
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            })
            await send({
                "type": "http.response.body",
                "body": '{"error": "Empty stream"}\n'.encode("utf-8"),
                "more_body": True,
            })
        await send({
            "type": "http.response.body",
            "body": b"",
            "more_body": False,
        })

        if self.background is not None:
            await self.background()


async def _annotation_async_gen(request: Request, model_service: AbstractModelService) -> AsyncGenerator:
    try:
        buffer = ""
        doc_idx = 0
        async for chunk in request.stream():
            decoded = chunk.decode("utf-8")
            if not decoded:
                break
            buffer += decoded
            while "\n" in buffer:
                lines = buffer.split("\n")
                line = lines[0]
                buffer = "\n".join(lines[1:]) if len(lines) > 1 else ""
                if line.strip():
                    try:
                        json_line_obj = json.loads(line)
                        TextStreamItem(**json_line_obj)
                        annotations = await model_service.annotate_async(json_line_obj["text"])
                        for anno in annotations:
                            anno.doc_name = json_line_obj.get("name", str(doc_idx))
                            yield anno.json(exclude_none=True) + "\n"
                    except json.JSONDecodeError:
                        yield json.dumps({"error": "Invalid JSON Line", "content": line}) + "\n"
                    except ValidationError:
                        yield json.dumps({"error": f"Invalid JSON properties found. The schema should be {TextStreamItem.schema_json()}", "content": line}) + "\n"
                    finally:
                        doc_idx += 1
        if buffer.strip():
            try:
                json_line_obj = json.loads(buffer)
                TextStreamItem(**json_line_obj)
                annotations = model_service.annotate(json_line_obj["text"])
                for anno in annotations:
                    anno.doc_name = json_line_obj.get("name", str(doc_idx))
                    yield anno.json(exclude_none=True) + "\n"
            except json.JSONDecodeError:
                yield json.dumps({"error": "Invalid JSON Line", "content": buffer}) + "\n"
            except ValidationError:
                yield json.dumps({"error": f"Invalid JSON properties found. The schema should be {TextStreamItem.schema_json()}", "content": buffer}) + "\n"
            finally:
                doc_idx += 1
    except ClientDisconnect:
        logger.debug("Client disconnected while annotations were being streamed")
