"""cms-mcp-server"""

import os
import uvicorn
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.middleware import Middleware
from mcp.server.fastmcp import FastMCP
from cms_client import (
    ApiClient,
    Configuration,
    MetadataApi,
    AnnotationsApi,
    RedactionApi,
    TrainingApi,
)
from app.mcp.domain import AppContext, TransportType
from app.mcp.tools import metadata, annotation, train_eval
from app.mcp.oauth.oauth import OAuthManager
from app.mcp.oauth.middleware import OAuthMiddleware
from app.mcp.logger import get_logger

logger = get_logger(__name__)

COGSTACK_MODELSERVE_INSTRUCTIONS = """
You are the CogStack ModelServe assistant. You expose MCP tools to interact with deployed models via the MCP server interface. Your job is to interpret the user's natural language requests and map them to the correct MCP tool calls with appropriate arguments. Use the following guidelines:

1. Interpret intent
    - If the user asks about model information, status, or metadata (e.g. "what model is running?", "model details"), use the `get_model_info` tool.
    - If the user wants to annotate text or extract entities (e.g. "annotate this text", "find entities in this document"), use the `get_annotations` tool.
    - If the user wants to redact sensitive information (e.g. "redact this text", "remove personal information"), use the `redact_text` tool.
    - If the user asks about training or evaluation status (e.g. "what's the status of training job X?", "evaluation info for job Y"), use the `get_train_eval_info` tool.
    - If the user wants training or evaluation metrics (e.g. "get metrics for training job X", "evaluation metrics for job Y"), use the `get_train_eval_metrics` tool.

2. Validate arguments
    - For `get_model_info`: No arguments required.
    - For `get_annotations`: Requires text input (string).
    - For `redact_text`: Requires text input (string).
    - For `get_train_eval_info`: Requires train_eval_id (string).
    - For `get_train_eval_metrics`: Requires train_eval_id (string).
    - Ensure inputs are provided and are valid strings.
    - If required arguments are missing, ask the user to provide them.

3. Form an MCP tool call
    - Use the proper tool name and arguments according to the available tools.
    - Example calls:
        - get_model_info() - no arguments
        - get_annotations(text="Patient has fever and chest pain")
        - redact_text(text="Patient John Doe has fever")
        - get_train_eval_info(train_eval_id="job123")
        - get_train_eval_metrics(train_eval_id="job123")

4. Handle responses
    - Return the tool output (model info, annotations, redacted text, training/evaluation status, metrics) to the user in human-friendly form.
    - If the tool returns an error, capture it and present a clear error message (e.g., "Failed to get model info: <error details>").

5. Fallback / clarification
    - If the user's request is ambiguous (e.g. "process this text" without specifying annotation or redaction), ask for clarification: "Do you want to annotate the text or redact sensitive information?"
    - If no matching tool exists for the request, respond: "I'm sorry, I don't have that capability in the CogStack ModelServe interface."

6. Security / permissions check
    - Only use the available tools: get_model_info, get_annotations, redact_text, get_train_eval_info, get_train_eval_metrics.
    - Don't allow arbitrary operations outside these tools.

Available tools:
    - get_model_info(): Gets information about the running model (no arguments)
    - get_annotations(text: str): Gets annotations for the provided text using the running model
    - redact_text(text: str): Redacts extracted entities from the provided text using the running model
    - get_train_eval_info(train_eval_id: str): Gets training or evaluation status for the specified job ID
    - get_train_eval_metrics(train_eval_id: str): Gets training or evaluation metrics for the specified job ID
"""

# Global OAuth manager
oauth_manager: Optional[OAuthManager] = None


@asynccontextmanager
async def app_lifespan(app: FastMCP) -> AsyncIterator[AppContext]:
    """Application lifespan context manager"""
    configuration = Configuration(
        host=os.environ.get("CMS_BASE_URL", "http://127.0.0.1:8000")
    )
    configuration.access_token = os.environ.get("CMS_ACCESS_TOKEN", "")
    configuration.api_key["APIKeyCookie"] = os.environ.get("CMS_API_KEY", "Bearer")

    api_client = ApiClient(configuration)
    try:
        yield AppContext(
            metadata_api=MetadataApi(api_client),
            annotation_api=AnnotationsApi(api_client),
            redaction_api=RedactionApi(api_client),
            training_api=TrainingApi(api_client),
        )
    finally:
        pass  # api_client.close() is only required when using httpx.AsyncClient

def create_server() -> Starlette:
    """Create the main Starlette application with MCP and optional OAuth"""
    global oauth_manager

    host = os.environ.get("CMS_MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("CMS_MCP_SERVER_PORT", "8080"))

    mcp_server = FastMCP(
        "cms_mcp_server",
        COGSTACK_MODELSERVE_INSTRUCTIONS,
        lifespan=app_lifespan,
        host=host,
        port=port,
    )

    metadata.register_module(mcp_server)
    annotation.register_module(mcp_server)
    train_eval.register_module(mcp_server)

    routes = []
    middleware = []
    oauth_enabled = os.environ.get("CMS_MCP_OAUTH_ENABLED", "false").lower() == "true"

    if oauth_enabled:
        try:
            base_url = f"http://{host}:{port}"
            oauth_manager = OAuthManager(base_url)
            oauth_routes = oauth_manager.create_oauth_routes()
            routes.extend(oauth_routes)

            middleware.append(
                Middleware(
                    OAuthMiddleware,
                    oauth_manager=oauth_manager,
                    public_paths=[
                        "/oauth/",
                        "/docs",
                        "/openapi.json",
                        "/redoc",
                        "/health",
                        "/.well-known/",
                    ]
                )
            )

            logger.info("OAuth authentication enabled")
        except Exception as e:
            logger.error(f"Failed to setup OAuth: {e}")
            logger.warning("Server will run without OAuth authentication")

    if os.environ.get("CMS_MCP_TRANSPORT") == TransportType.SSE.value:
        routes.append(Mount("/", app=mcp_server.sse_app("")))
        logger.info(f"MCP SSE endpoint mounted at http://{host}:{port}/sse")
    elif os.environ.get("CMS_MCP_TRANSPORT") == TransportType.STREAMABLE_HTTP.value:
        # OAuth is not supported yet for HTTP transport
        logger.warning("OAuth disabled for HTTP transport due to FastMCP limitations")
        logger.info(f"MCP HTTP endpoint mounted at http://{host}:{port}/mcp")
        return mcp_server.streamable_http_app()
    elif os.environ.get("CMS_MCP_TRANSPORT") == TransportType.STDIO.value:
        logger.info("MCP running in STDIO mode")
        mcp_server.run(transport=TransportType.STDIO.value)
    else:
        raise ValueError(f"Unsupported transport type: {os.environ.get('CMS_MCP_TRANSPORT')}")

    app = Starlette(
        routes=routes,
        middleware=middleware,
    )

    return app

def main() -> None:
    host = os.environ.get("CMS_MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("CMS_MCP_SERVER_PORT", "8080"))

    app = create_server()

    if os.environ.get("CMS_MCP_OAUTH_ENABLED", "false").lower() == "true" and os.environ.get("CMS_MCP_TRANSPORT") != TransportType.STREAMABLE_HTTP.value:
        logger.info(f"OAuth login: http://{host}:{port}/oauth/login")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("FASTMCP_LOG_LEVEL", "info").lower(),
    )

server: FastMCP | None = None

if os.environ.get("CMS_MCP_DEV", "0") == "1":
    server = FastMCP(
        "cms_mcp_server",
        COGSTACK_MODELSERVE_INSTRUCTIONS,
        lifespan=app_lifespan,
        host=os.environ.get("CMS_MCP_SERVER_HOST", "127.0.0.1"),
        port=int(os.environ.get("CMS_MCP_SERVER_PORT", "8080")),
    )
    metadata.register_module(server)
    annotation.register_module(server)
    train_eval.register_module(server)

if __name__ == "__main__":
    main()
