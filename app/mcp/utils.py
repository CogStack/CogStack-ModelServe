import os
from typing import Callable, Any, Awaitable, Union
from functools import wraps


def require_api_key(tool_fn: Callable[..., Union[Any, Awaitable[Any]]]) -> Callable[..., Union[Any, Awaitable[Any]]]:
    @wraps(tool_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Union[Any, Awaitable[Any]]:
        ctx = kwargs["ctx"] if kwargs and "ctx" in kwargs else None
        if ctx and hasattr(ctx, "request_context"):
            api_key = (
                ctx.request_context.request.headers.get("x-api-key", None) or
                ctx.request_context.request.headers.get("X-API-Key", None) or
                ctx.request_context.request.headers.get("X-Api-Key", None)
            )
        else:
            api_key = None
        api_keys_env = os.environ.get("MCP_API_KEYS")

        if not api_keys_env:    # No API-key-based authentication required
            return tool_fn(*args, **kwargs)

        allowed_keys = [key.strip() for key in api_keys_env.split(",")]

        if not api_key or api_key not in allowed_keys:
            raise PermissionError("The API key is either invalid or missing")

        return tool_fn(*args, **kwargs)
    return wrapper
