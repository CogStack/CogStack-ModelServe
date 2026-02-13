import os
import app.mcp
from unittest.mock import patch
from starlette.applications import Starlette
from app.mcp.server import create_server, COGSTACK_MODELSERVE_INSTRUCTIONS


class TestMpcServer:

    @patch.dict(os.environ, {"CMS_MCP_TRANSPORT": "http"}, clear=True)
    def test_create_server(self):
        app = create_server()
        assert app is not None
        assert isinstance(app, Starlette)

    def test_server_version(self):
        assert hasattr(app.mcp, "__version__")
        assert isinstance(app.mcp.__version__, str)

    @patch.dict(
        os.environ,
        {"CMS_MCP_API_KEYS": "key1,key2,key3", "CMS_MCP_TRANSPORT": "http"},
        clear=True
    )
    def test_create_server_with_api_keys(self):
        app = create_server()
        assert app is not None
        assert os.environ.get("CMS_MCP_API_KEYS") == "key1,key2,key3"

    @patch.dict(os.environ, {"CMS_MCP_TRANSPORT": "http"}, clear=True)
    def test_create_server_without_api_keys(self):
        app = create_server()
        assert app is not None
        assert os.environ.get("CMS_MCP_API_KEYS") is None
