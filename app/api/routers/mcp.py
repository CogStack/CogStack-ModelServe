import logging
from typing import List

from fastapi import APIRouter, Depends
from mcp.server.fastmcp import FastMCP

import app.api.globals as cms_globals
from app.domain import Annotation
from app.model_services.base import AbstractModelService

logger = logging.getLogger("cms")

# Create FastMCP server instance
mcp_server = FastMCP(
    name="CogStack ModelServe MCP Server",
    instructions="A model serving system for CogStack NLP solutions that provides text annotation capabilities."
)

# Create FastAPI router
router = APIRouter()

# Global model service instance - will be set during router initialization
_model_service: AbstractModelService = None

# Add the annotate tool to the MCP server
@mcp_server.tool()
async def annotate_text(text: str) -> List[dict]:
    """
    Annotate text using the configured model service.

    This tool extracts named entities and other annotations from the provided text
    using the CogStack ModelServe annotation pipeline.

    Args:
        text: The text to be annotated

    Returns:
        A list of annotation dictionaries containing the extracted entities
    """
    global _model_service
    if _model_service is None:
        # Get model service from global dependency
        _model_service = cms_globals.model_service_dep()

    try:
        annotations = _model_service.annotate(text)
        # Convert Annotation objects to dictionaries for JSON serialization
        return [annotation.dict() for annotation in annotations]
    except Exception as e:
        logger.error(f"Error annotating text: {e}")
        raise

# Note: For full MCP protocol support, you would mount the FastMCP app here.
# For simplicity, we're providing direct HTTP endpoints for tool operations.

# For HTTP-based MCP transport, provide a simple tool calling endpoint
@router.post("/mcp/tools/call")
async def call_tool(request: dict):
    """
    Direct tool call endpoint for HTTP-based MCP clients.

    This allows HTTP clients to call the annotate tool directly.
    """
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})

        if tool_name == "annotate_text":
            text = arguments.get("text")
            if not text:
                return {"error": "Missing 'text' argument"}

            annotations = await annotate_text(text)
            return {
                "result": annotations,
                "success": True
            }
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return {"error": str(e), "success": False}

@router.get("/mcp/tools")
async def list_tools():
    """
    List available MCP tools.
    """
    return {
        "tools": [
            {
                "name": "annotate_text",
                "description": "Annotate text using the configured model service",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to be annotated"
                        }
                    },
                    "required": ["text"]
                }
            }
        ]
    }