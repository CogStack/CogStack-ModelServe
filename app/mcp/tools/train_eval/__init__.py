from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from .get_train_eval_info import get_train_eval_info
from .get_train_eval_metrics import get_train_eval_metrics
from ...domain import AppContext
from ...utils import require_api_key


def register_module(mcp: FastMCP) -> None:

    @mcp.tool(
        name="get_train_eval_info",
        description="Gets training or evaluation status of the running model.",
    )
    @require_api_key
    def get_train_eval_info_tool(train_eval_id: str, ctx: Context[ServerSession, AppContext]) -> dict:
        """
        Gets training or evaluation status of the running model.

        Args:
            train_eval_id (str): The ID of the training or evaluation job.
            ctx (Context[ServerSession, AppContext]): The application context.

        Returns:
            dict: A dictionary containing the training or evaluation status.
        """
        training_api = ctx.request_context.lifespan_context.training_api
        return get_train_eval_info(train_eval_id, training_api)

    @mcp.tool(
        name="get_train_eval_metrics",
        description="Gets training or evaluation metrics.",
    )
    @require_api_key
    def get_train_eval_metrics_tool(train_eval_id: str, ctx: Context[ServerSession, AppContext]) -> dict:
        """
        Gets training or evaluation metrics.

        Args:
            train_eval_id (str): The ID of the training or evaluation job.
            ctx (Context[ServerSession, AppContext]): The application context.

        Returns:
            dict: A dictionary containing the training or evaluation metrics.
        """
        training_api = ctx.request_context.lifespan_context.training_api
        return get_train_eval_metrics(train_eval_id, training_api)
