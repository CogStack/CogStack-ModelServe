import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from app.mcp.oauth.middleware import OAuthMiddleware, get_oauth_token_from_request, require_auth
from app.mcp.oauth.oauth import OAuthManager, OAuthToken

@pytest.fixture
def mock_oauth_manager():
    manager = Mock(spec=OAuthManager)
    return manager

@pytest.fixture
def oauth_token():
    return OAuthToken(
        access_token="test_access_token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="test_refresh_token",
    )
@pytest.fixture
def mock_app():
    async def mock_scope(scope, receive, send):
        pass
    return mock_scope

class TestOAuthMiddleware:
    def test_init_with_public_paths(self, mock_app, mock_oauth_manager):
        public_paths = ["/oauth/", "/docs", "/health"]
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=public_paths
        )

        assert middleware.oauth_manager == mock_oauth_manager
        assert middleware.public_paths == public_paths

    def test_init_without_public_paths(self, mock_app, mock_oauth_manager):
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager
        )

        assert middleware.oauth_manager == mock_oauth_manager
        assert middleware.public_paths == []

    def test_is_public_path_with_match(self, mock_app, mock_oauth_manager):
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/", "/docs", "/health"]
        )

        assert middleware._is_public_path("/oauth/login") is True
        assert middleware._is_public_path("/docs") is True
        assert middleware._is_public_path("/health") is True

    def test_is_public_path_without_match(self, mock_app, mock_oauth_manager):
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/", "/docs", "/health"]
        )

        assert middleware._is_public_path("/api/endpoint") is False
        assert middleware._is_public_path("/protected/path") is False
        assert middleware._is_public_path("/users/profile") is False

    @pytest.mark.asyncio
    async def test_dispatch_sse_bypasses_auth(self, mock_app, mock_oauth_manager):
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/sse"
        request.cookies = {}
        request.headers = {}
        call_next = AsyncMock(return_value=Mock())
        
        await middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_dispatch_public_path_bypasses_auth(self, mock_app, mock_oauth_manager):
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/", "/docs", "/health"]
        )
        request = Mock(spec=Request)
        request.url.path = "/oauth/login"
        request.cookies = {}
        request.headers = {}
        call_next = AsyncMock(return_value=Mock())
        
        await middleware.dispatch(request, call_next)
        
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_dispatch_no_session_returns_401(self, mock_app, mock_oauth_manager):
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.cookies = {}
        request.headers = {}
        request.query_params = {}
        call_next = AsyncMock(return_value=Mock())
        
        response = await middleware.dispatch(request, call_next)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_session_cookie_authenticated(self, mock_app, mock_oauth_manager, oauth_token):
        mock_oauth_manager.get_valid_token = AsyncMock(return_value=oauth_token)
        
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.cookies = {"cms_mcp_session": "valid_session_id"}
        request.headers = {}
        request.query_params = {}
        request.state = Mock()
        call_next = AsyncMock(return_value=Mock())
        
        await middleware.dispatch(request, call_next)
        
        mock_oauth_manager.get_valid_token.assert_called_once_with("valid_session_id")
        call_next.assert_called_once_with(request)
        assert request.state.oauth_token == oauth_token
        assert request.state.session_id == "valid_session_id"

    @pytest.mark.asyncio
    async def test_dispatch_bearer_token_authenticated(self, mock_app, mock_oauth_manager, oauth_token):
        mock_oauth_manager.get_valid_token = AsyncMock(return_value=oauth_token)
        
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.cookies = {}
        request.headers = {"Authorization": "Bearer valid_token"}
        request.query_params = {}
        request.state = Mock()
        call_next = AsyncMock(return_value=Mock())
        
        await middleware.dispatch(request, call_next)
        
        mock_oauth_manager.get_valid_token.assert_called_once_with("valid_token")
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_dispatch_query_param_session_authenticated(self, mock_app, mock_oauth_manager, oauth_token):
        mock_oauth_manager.get_valid_token = AsyncMock(return_value=oauth_token)
        
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.cookies = {}
        request.headers = {}
        request.query_params = {"session_id": "valid_session_id"}
        request.state = Mock()
        call_next = AsyncMock(return_value=Mock())
        
        await middleware.dispatch(request, call_next)
        
        mock_oauth_manager.get_valid_token.assert_called_once_with("valid_session_id")
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_dispatch_invalid_session_returns_401(self, mock_app, mock_oauth_manager):
        """Test that invalid/expired session returns 401"""
        mock_oauth_manager.get_valid_token = AsyncMock(return_value=None)
        
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.cookies = {"cms_mcp_session": "invalid_session_id"}
        request.headers = {}
        request.query_params = {}
        request.state = Mock()
        call_next = AsyncMock(return_value=Mock())
        
        response = await middleware.dispatch(request, call_next)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_exception_returns_500(self, mock_app, mock_oauth_manager):
        mock_oauth_manager.get_valid_token = AsyncMock(side_effect=Exception("Database error"))
        
        middleware = OAuthMiddleware(
            app=mock_app,
            oauth_manager=mock_oauth_manager,
            public_paths=["/oauth/"]
        )
        request = Mock(spec=Request)
        request.url.path = "/api/protected"
        request.cookies = {"cms_mcp_session": "valid_session_id"}
        request.headers = {}
        request.query_params = {}
        request.state = Mock()
        call_next = AsyncMock(return_value=Mock())
        
        response = await middleware.dispatch(request, call_next)
        
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

class TestGetOAuthTokenFromRequest:

    def test_with_oauth_token(self, oauth_token):
        request = Mock(spec=Request)
        request.state.oauth_token = oauth_token
        
        token = get_oauth_token_from_request(request)
        
        assert token == "test_access_token"

    def test_without_oauth_token(self):
        request = Mock(spec=Request)
        del request.state.oauth_token
        
        token = get_oauth_token_from_request(request)
        
        assert token is None

class TestRequireAuth:

    def test_with_valid_token(self, oauth_token):
        request = Mock(spec=Request)
        request.state.oauth_token = oauth_token
        
        token = require_auth(request)
        
        assert token == "test_access_token"

    def test_without_token_raises_401(self):
        
        request = Mock(spec=Request)
        del request.state.oauth_token
        
        with pytest.raises(HTTPException) as exc_info:
            require_auth(request)
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"
