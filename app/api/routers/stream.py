import json
import logging
import asyncio
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.websockets import WebSocketDisconnect
from starlette.requests import ClientDisconnect

import app.api.globals as cms_globals

from typing import Any, Mapping, Optional, AsyncGenerator
from typing_extensions import Annotated
from starlette.types import Receive, Scope, Send
from starlette.background import BackgroundTask
from starlette.status import HTTP_202_ACCEPTED
from fastapi import APIRouter, Depends, Query, Request, Response, WebSocket, WebSocketException, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError, BaseModel
from app.domain import Tags, TextStreamItem
from app.model_services.base import AbstractModelService
from app.utils import get_settings
from app.api.utils import get_rate_limiter
from app.api.auth.users import get_user_manager, CmsUserManager

PATH_STREAM_PROCESS = "/process"
PATH_WS = "/ws"
PATH_SSE_EVENTS = "/sse/events"
PATH_SSE_PROCESS = "/sse/process"
SSE_CONNECTION_TIMEOUT_SECONDS = 300
SSE_CONNECTION_MAX_RETRIES = 10


router = APIRouter()
config = get_settings()
limiter = get_rate_limiter(config)
sse_clients: dict[str, asyncio.Queue] = {}
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


@router.get(
    PATH_WS,
    tags=[Tags.Annotations.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
    description="WebSocket info endpoint for real-time NER entity extraction. Use ws://host:port/stream/ws to establish an actual WebSocket connection.",
    include_in_schema=True,
)
async def get_inline_entities_from_websocket_info() -> "_WebSocketInfo":
    """
    Information about the WebSocket endpoint for real-time NER entity extraction.

    This endpoint provides documentation for the WebSocket connection available at the same path.
    Connect to ws://host:port/stream/ws and send texts to retrieve annotated results.
    """
    return _WebSocketInfo()


@router.websocket(PATH_WS)
@limiter.exempt
async def get_inline_entities_from_websocket(
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
            jwt_backend = cms_globals.props.auth_backends[0]    # type: ignore
            cookie_backend = cms_globals.props.auth_backends[1] # type: ignore
            auth_header = websocket.headers.get("Authorization", "")
            cookie = websocket.cookies.get("fastapiusersauth", "")

            if not auth_header and not cookie:
                raise WebSocketException(
                    code=WS_1008_POLICY_VIOLATION,
                    reason="Authentication credentials not found (Bearer token or cookie required)",
                )

            user = None
            try:
                if auth_header:
                    bearer_token = auth_header.split(" ", 1)[1].strip()
                    user = await jwt_backend.get_strategy().read_token(bearer_token, user_manager)  # type: ignore
                else:
                    user = await cookie_backend.get_strategy().read_token(cookie, user_manager) # type: ignore
            except Exception:
                raise WebSocketException(
                    code=WS_1008_POLICY_VIOLATION,
                    reason="Invalid authentication credential)",
                )
            else:
                if user is None or not user.is_active:
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


@router.get(PATH_SSE_EVENTS)
@limiter.exempt
async def get_entities_stream_from_sse(
    request: Request,
    client_id: Annotated[str, Query(description="Unique client identifier for the SSE connection")],
    keep_alive: Annotated[Optional[bool], Query(description="Whether to keep the conneciton alive after periods of inactivity")] = False,
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint to receive NER entities as stream events for a specific client.

    Args:
        request (Request): The request object.
        client_id (str): The unique client identifier for the SSE connection.
        keep_alive (Optional[bool]): Whether to keep the connection alive after periods of inactivity.

    Returns:
        StreamingResponse: A streaming response for the SSE connection.
    """
    if client_id in sse_clients and sse_clients[client_id] is not None:
        queue = sse_clients[client_id]
    else:
        queue = asyncio.Queue()
        sse_clients[client_id] = queue

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield ": connected\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    logger.debug(f"Waiting for event for client {client_id}")
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_CONNECTION_TIMEOUT_SECONDS)

                    if isinstance(event, dict) and event.get("_control") == "close":
                        logger.debug(f"Closing SSE for client {client_id} as requested")
                        break

                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    if keep_alive:
                        logger.debug(f"Sending keepalive for client {client_id} after timeout")
                        yield ": keepalive\n\n"
                        continue
                    else:
                        logger.debug(f"Timeout reached for client {client_id}, closing connection")
                        break
        except asyncio.CancelledError:
            logger.debug(f"SSE connection for client {client_id} cancelled")
        except Exception as e:
            logger.error(f"SSE error for client {client_id}: {e}")
            yield f"data: {json.dumps({'error': 'stream_error', 'message': str(e)})}\n\n"
        finally:
            sse_clients.pop(client_id, None)
            logger.debug(f"SSE disconnected for client {client_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post(
    PATH_SSE_PROCESS,
    tags=[Tags.Annotations.name],
    dependencies=[Depends(cms_globals.props.current_active_user)],
)
@limiter.exempt
async def send_text_jsonlines_for_processing(
    request: Request,
    client_id: Annotated[str, Query(description="Unique client identifier for the SSE connection")],
    model_service: AbstractModelService = Depends(cms_globals.model_service_dep),
) -> JSONResponse:
    """
    Sends texts in the JSON Lines format for processing and extracted NER entities will be received via Server-Sent Events (SSE).

    Args:
        request (Request): The request object containing the texts in JSON Lines.
        client_id (str): The unique client identifier for the SSE connection.
        model_service (AbstractModelService): The model service dependency.

    Returns:
        JSONResponse: A JSON response indicating the status of the request.
    """
    for _ in range(SSE_CONNECTION_MAX_RETRIES):
        queue = sse_clients.get(client_id)
        if queue:
            break
        await asyncio.sleep(0.1)
    else:
        raise HTTPException(status_code=400, detail="Client not connected. Please establish SSE connection first.")

    async def process_text(queue: asyncio.Queue, doc_name: str, text: str) -> None:
        try:
            await queue.put({"status": "started", "doc_name": doc_name, "text": text})
            annotations = await model_service.annotate_async(text)
            await asyncio.sleep(0.1)
            for anno in annotations:
                anno.doc_name = doc_name
                await queue.put({"type": "annotation", "data": anno.dict(exclude_none=True)})
            await queue.put({"status": "completed", "doc_name": doc_name})
        except asyncio.CancelledError:
            logger.debug(f"Processing for document {doc_name} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing document {doc_name}: {e}")
            await queue.put({"status": "error", "error": str(e), "doc_name": doc_name})

    tasks = []
    buffer = ""
    doc_idx = 0

    try:
        async for chunk in request.stream():
            decoded = chunk.decode("utf-8")
            if not decoded:
                break
            buffer += decoded

            while "\n" in buffer:
                newline_idx = buffer.index("\n")
                line = buffer[:newline_idx]
                buffer = buffer[newline_idx + 1:]

                if line.strip():
                    try:
                        json_line_obj = json.loads(line)
                        TextStreamItem(**json_line_obj)
                        task = asyncio.create_task(
                            process_text(
                                queue,
                                text=json_line_obj["text"],
                                doc_name=json_line_obj.get("name", f"doc_{doc_idx}"),
                            )
                        )
                        tasks.append(task)
                    except json.JSONDecodeError as e:
                        await queue.put({'status': 'error', 'error': f'Invalid JSON Line: {str(e)}', 'content': line})
                    except ValidationError as e:
                        await queue.put({'status': 'error', 'error': f'Invalid JSON properties: {str(e)}', 'content': line})
                    finally:
                        doc_idx += 1

        if buffer.strip():
            try:
                json_line_obj = json.loads(buffer)
                TextStreamItem(**json_line_obj)
                task = asyncio.create_task(
                    process_text(
                        queue,
                        text=json_line_obj["text"],
                        doc_name=json_line_obj.get("name", f"doc_{doc_idx}"),
                    )
                )
                tasks.append(task)
            except (json.JSONDecodeError, ValidationError) as e:
                await queue.put({'status': 'error', 'error': str(e), 'content': buffer})

    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        await queue.put({"status": "all_completed", "total_docs": doc_idx})

    return JSONResponse(content={"status": "accepted", "total_docs": doc_idx}, status_code=HTTP_202_ACCEPTED)

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


class _WebSocketInfo(BaseModel):
    message: str = "WebSocket endpoint for real-time NER entity extraction"
    example: str = """<form action="" onsubmit="send_doc(event)">
    <input type="text" id="cms-input" autocomplete="off"/>
    <button>Send</button>
</form>
<ul id="cms-output"></ul>
<script>
    var ws = new WebSocket("ws://localhost:8000/stream/ws");
    ws.onmessage = function(event) {
        document.getElementById("cms-output").appendChild(
            Object.assign(document.createElement('li'), { textContent: event.data })
        );
    };
    function send_doc(event) {
        ws.send(document.getElementById("cms-input").value);
        event.preventDefault();
    };
</script>"""
    protocol: str = "WebSocket"


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
