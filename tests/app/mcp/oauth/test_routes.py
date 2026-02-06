"""Tests for OAuth routes"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.mcp.oauth.routes import register_oauth_routes
from app.mcp.oauth.oauth import OAuthManager, OAuthToken


@pytest.fixture
def app():
    """Create a test FastAPI app"""
    return FastAPI()


@pytest.fixture
def oauth_manager():
    """Create an OAuthManager instance"""
    return OAuthManager(base_url="http://localhost:8080")


@pytest.fixture
def test_client(app, oauth_manager):
    """Create a test client with OAuth routes registered"""
    register_oauth_routes(app, oauth_manager)
    return TestClient(app)


class TestOAuthRoutes:
    """Tests for OAuth route handlers"""

    def test_oauth_login_returns_html(self, test_client):
        """Test that /oauth/login returns HTML content"""
        response = test_client.get("/oauth/login")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "CMS MCP Server" in response.text
        assert "Continue with Google" in response.text
        assert "Continue with GitHub" in response.text

    def test_oauth_authorize_unknown_provider(self, test_client):
        """Test that unknown provider returns 400"""
        response = test_client.get("/oauth/authorize/unknown_provider")
        
        assert response.status_code == 400
        assert "Unknown provider" in response.text

    def test_oauth_authorize_google_redirects(self, test_client):
        """Test that Google authorization redirects"""
        response = test_client.get("/oauth/authorize/google", follow_redirects=False)
        
        assert response.status_code == 307  # Temporary redirect
        assert response.headers["location"].startswith("https://accounts.google.com")

    def test_oauth_authorize_github_redirects(self, test_client):
        """Test that GitHub authorization redirects"""
        response = test_client.get("/oauth/authorize/github", follow_redirects=False)
        
        assert response.status_code == 307
        assert response.headers["location"].startswith("https://github.com/login/oauth/authorize")

    def test_oauth_callback_missing_code_or_state(self, test_client):
        """Test that missing code or state returns 400"""
        response = test_client.get("/oauth/callback/google")
        
        assert response.status_code == 400
        assert "Missing code or state" in response.text

    def test_oauth_callback_unknown_provider(self, test_client):
        """Test that unknown provider returns 400"""
        response = test_client.get(
            "/oauth/callback/unknown_provider",
            params={"code": "test_code", "state": "test_state"}
        )
        
        assert response.status_code == 400
        assert "Unknown provider" in response.text

    def test_oauth_status_no_session(self, test_client):
        """Test that /oauth/status returns login page when no session"""
        response = test_client.get("/oauth/status")
        
        assert response.status_code == 200
        assert "No Active Session" in response.text or "not authenticated" in response.text.lower()

    def test_oauth_logout_redirects(self, test_client):
        """Test that /oauth/login redirects to login page"""
        response = test_client.get("/oauth/logout", follow_redirects=False)
        
        assert response.status_code == 307
        assert response.headers["location"] == "/oauth/login"


class TestRegisterOAuthRoutes:
    """Tests for register_oauth_routes function"""

    def test_register_routes_adds_all_endpoints(self, app, oauth_manager):
        """Test that all OAuth routes are registered"""
        register_oauth_routes(app, oauth_manager)
        
        # Check that routes are added
        route_paths = [route.path for route in app.routes]
        assert "/oauth/login" in route_paths
        assert "/oauth/authorize/{provider}" in route_paths
        assert "/oauth/callback/{provider}" in route_paths
        assert "/oauth/status" in route_paths
        assert "/oauth/logout" in route_paths

    def test_register_routes_with_different_oauth_manager(self, app):
        """Test registering routes with a custom OAuth manager"""
        custom_manager = OAuthManager(base_url="http://custom:9000")
        register_oauth_routes(app, custom_manager)
        
        # Verify the routes were added
        route_paths = [route.path for route in app.routes]
        assert "/oauth/login" in route_paths
        assert "/oauth/authorize/{provider}" in route_paths


class TestOAuthCallbackWithMockedProvider:
    """Tests for OAuth callback with mocked provider"""

    @pytest.fixture
    def mock_oauth_manager(self):
        """Create a mock OAuth manager"""
        manager = Mock(spec=OAuthManager)
        return manager

    @pytest.fixture
    def client_with_mock(self, app, mock_oauth_manager):
        """Create a test client with mocked OAuth manager"""
        register_oauth_routes(app, mock_oauth_manager)
        return TestClient(app)

    def test_oauth_callback_with_error(self, client_with_mock):
        """Test OAuth callback handles errors"""
        response = client_with_mock.get(
            "/oauth/callback/google",
            params={"error": "access_denied", "code": None, "state": None}
        )
        
        assert response.status_code == 400
        assert "Authentication Error" in response.text or "Authentication Failed" in response.text

    def test_oauth_callback_state_mismatch(self, app, mock_oauth_manager, client_with_mock):
        """Test OAuth callback rejects state mismatch"""
        # Setup mock to return a provider
        mock_provider = Mock()
        mock_provider.generate_authorization_url.return_value = ("https://auth.url", "correct_state")
        mock_provider.verify_state.return_value = True
        mock_oauth_manager.get_provider.return_value = mock_provider
        
        response = client_with_mock.get(
            "/oauth/callback/google",
            params={"code": "test_code", "state": "wrong_state"},
            cookies={"oauth_state_google": "correct_state"}
        )
        
        assert response.status_code == 400
        assert "Invalid state" in response.text

    def test_oauth_callback_verification_failure(self, app, mock_oauth_manager, client_with_mock):
        """Test OAuth callback fails when state verification fails"""
        mock_provider = Mock()
        mock_provider.verify_state.return_value = False
        mock_oauth_manager.get_provider.return_value = mock_provider
        
        response = client_with_mock.get(
            "/oauth/callback/google",
            params={"code": "test_code", "state": "test_state"},
            cookies={"oauth_state_google": "test_state"}
        )
        
        assert response.status_code == 400
        assert "State verification failed" in response.text
