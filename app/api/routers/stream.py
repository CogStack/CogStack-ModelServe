import json
import logging
import asyncio
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.websockets import WebSocketDisconnect
from starlette.requests import ClientDisconnect

import api.globals as cms_globals

from typing import Any, Mapping, Optional, AsyncGenerator
from starlette.types import Receive, Scope, Send
from starlette.background import BackgroundTask
from fastapi import APIRouter, Depends, Request, Response, WebSocket, WebSocketException
from pydantic import ValidationError
from domain import Annotation, Tags, TextStreamItem
from model_services.base import AbstractModelService
from utils import get_settings
from api.utils import get_rate_limiter
from api.auth.users import get_user_manager, CmsUserManager

PATH_STREAM_PROCESS = "/process"
PATH_WS_PROCESS = "/ws"

router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)
logger = logging.getLogger("cms")


@router.post(PATH_STREAM_PROCESS,
             tags=[Tags.Annotations.name],
             dependencies=[Depends(cms_globals.props.current_active_user)],
             description="Extract the NER entities from a stream of texts in the JSON Lines format")
@limiter.limit(config.PROCESS_BULK_RATE_LIMIT)
async def get_entities_stream_from_jsonlines_stream(request: Request,
                                                    model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> Response:
    annotation_stream = _annotation_async_gen(request, model_service)
    return _LocalStreamingResponse(annotation_stream, media_type="application/x-ndjson; charset=utf-8")


@router.websocket(PATH_WS_PROCESS)
# @limiter.limit(config.PROCESS_BULK_RATE_LIMIT)  # Not supported yet
async def get_inline_annotations_from_websocket(websocket: WebSocket,
                                                user_manager: CmsUserManager = Depends(get_user_manager),
                                                model_service: AbstractModelService = Depends(cms_globals.model_service_dep)) -> None:
    monitor_idle_task = None
    try:
        if get_settings().AUTH_USER_ENABLED == "true":
            cookie = websocket.cookies.get("fastapiusersauth")
            if cookie is None:
                raise WebSocketException(code=WS_1008_POLICY_VIOLATION, reason="Authentication cookie not found")
            user = await cms_globals.props.auth_backends[1].get_strategy().read_token(cookie, user_manager)
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
                annotations = await model_service.async_annotate(text)
                annotated_text = ""
                start_index = 0
                for annotation in annotations:
                    annotated_text += f'{text[start_index:annotation["start"]]}[{annotation["label_name"]}: {text[annotation["start"]:annotation["end"]]}]'
                    start_index = annotation["end"]
                annotated_text += text[start_index:]
            except Exception as e:
                await websocket.send_text(f"ERROR: {str(e)}")
            else:
                await websocket.send_text(annotated_text)
    except WebSocketDisconnect as e:
        logger.warning(str(e))
    finally:
        try:
            if monitor_idle_task is not None:
                monitor_idle_task.cancel()
            await websocket.close()
        except RuntimeError as e:
            logger.warning(str(e))


class _LocalStreamingResponse(Response):

    def __init__(self,
                 content: Any,
                 status_code: int = 200,
                 headers: Optional[Mapping[str, str]] = None,
                 media_type: Optional[str] = None,
                 background: Optional[BackgroundTask] = None) -> None:
        self.content = content
        self.status_code = status_code
        self.media_type = self.media_type if media_type is None else media_type
        self.background = background
        self.init_headers(headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        response_started = False
        max_chunk_size = 1024
        async for line in self.content:
            if not response_started:
                await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
                response_started = True
            line_bytes = line.encode("utf-8")
            for i in range(0, len(line_bytes), max_chunk_size):
                chunk = line_bytes[i:i + max_chunk_size]
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        if not response_started:
            await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
            await send({"type": "http.response.body", "body": '{"error": "Empty stream"}\n'.encode("utf-8"), "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

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
                        annotations = await model_service.async_annotate(json_line_obj["text"])
                        for annotation in annotations:
                            annotation["doc_name"] = json_line_obj.get("name", str(doc_idx))
                            yield Annotation(**annotation).json(exclude_none=True) + "\n"
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
                for annotation in annotations:
                    annotation["doc_name"] = json_line_obj.get("name", str(doc_idx))
                    yield Annotation(**annotation).json(exclude_none=True) + "\n"
            except json.JSONDecodeError:
                yield json.dumps({"error": "Invalid JSON Line", "content": buffer}) + "\n"
            except ValidationError:
                yield json.dumps({"error": f"Invalid JSON properties found. The schema should be {TextStreamItem.schema_json()}", "content": buffer}) + "\n"
            finally:
                doc_idx += 1
    except ClientDisconnect:
        logger.warning("Client disconnected while annotations were being streamed")
