from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from .get_model_info import model_info
from ...domain import AppContext
from ...utils import require_api_key


def register_module(mcp: FastMCP) -> None:

    @mcp.tool(
        name="get_model_info",
        description="Gets information about the running model.",
    )
    @require_api_key
    def get_model_info_tool(ctx: Context[ServerSession, AppContext]) -> dict:
        """
        Gets information about the running model.

        Args:
            ctx (Context[ServerSession, AppContext]): The application context.

        Returns:
            dict: A dictionary containing the model metadata information.
        """
        metadata_api = ctx.request_context.lifespan_context.metadata_api
        return model_info(metadata_api)
