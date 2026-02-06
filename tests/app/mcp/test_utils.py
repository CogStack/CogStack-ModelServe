import os
import pytest
from unittest.mock import Mock, patch
from app.mcp.utils import require_api_key


@pytest.mark.asyncio
async def test_require_api_key():

    @require_api_key
    async def mock_tool(anystr, ctx):
        return "success"

    with patch.dict(os.environ, {"MCP_API_KEYS": "key1,key2"}, clear=True):
        ctx = Mock()
        ctx.request_context.request.headers.get.return_value = "key1"
        result = await mock_tool("test", ctx=ctx)
        assert result == "success"

    with patch.dict(os.environ, {"MCP_API_KEYS": "key1,key2"}, clear=True):
        ctx = Mock()
        ctx.request_context.request.headers.get.return_value = "invalid"
        with pytest.raises(PermissionError):
            await mock_tool("test", ctx=ctx)

    with patch.dict(os.environ, {}, clear=True):
        ctx = Mock()
        result = await mock_tool("test", ctx=ctx)
        assert result == "success"
