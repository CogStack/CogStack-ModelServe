
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from .get_annotations import annotate
from .get_redaction import redact
from ...domain import AppContext
from ...utils import require_api_key


def register_module(mcp: FastMCP) -> None:

    @mcp.tool(
        name="get_annotations",
        description="Gets annotations for the provided text using the running model.",
    )
    @require_api_key
    def get_annotations_tool(text: str, ctx: Context[ServerSession, AppContext]) -> dict:
        """
        Gets annotations for the provided text using the running model.

        Args:
            text (str): The input text to be annotated.
            ctx (Context[ServerSession, AppContext]): The application context.

        Returns:
            dict: A dictionary containing the annotations.
        """
        annotation_api = ctx.request_context.lifespan_context.annotation_api
        return annotate(text, annotation_api)

    @mcp.tool(
        name="redact_text",
        description="Redacts extracted entities from the provided text using the running model.",
    )
    @require_api_key
    def get_redaction_tool(text: str, ctx: Context[ServerSession, AppContext]) -> dict:
        """
        Redacts extracted entities from the provided text using the running model.

        Args:
            text (str): The input text to be redacted.
            ctx (Context[ServerSession, AppContext]): The application context.

        Returns:
            dict: A dictionary containing the redacted text.
        """
        redaction_api = ctx.request_context.lifespan_context.redaction_api
        return redact(text, redaction_api)
