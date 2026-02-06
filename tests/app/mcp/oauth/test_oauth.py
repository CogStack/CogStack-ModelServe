import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from app.mcp.oauth.oauth import (
    OAuthConfig,
    OAuthProvider,
    GitHubOAuthProvider,
    GoogleOAuthProvider,
    OAuthManager,
    OAuthToken,
)


@pytest.fixture
def oauth_token():
    return OAuthToken(
        access_token="test_access_token",
        token_type="Bearer",
        expires_in=3600,
        refresh_token="test_refresh_token"
    )


@pytest.fixture
def oauth_config():
    return OAuthConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://example.com/auth",
        token_url="https://example.com/token",
        userinfo_url="https://example.com/userinfo",
        redirect_uri="http://localhost:8080/oauth/callback",
        scope="openid email profile"
    )


@pytest.fixture
def oauth_provider(oauth_config):
    return OAuthProvider(config=oauth_config)


@pytest.fixture
def oauth_manager():
    return OAuthManager(base_url="http://localhost:8080")


class TestOAuthProvider:

    def test_generate_authorization_url_without_state(self, oauth_provider):
        auth_url, state = oauth_provider.generate_authorization_url()
        
        assert auth_url.startswith("https://example.com/auth?")
        assert "client_id=test_client_id" in auth_url
        assert "redirect_uri=http://localhost:8080/oauth/callback" in auth_url
        assert "response_type=code" in auth_url
        assert "scope=openid email profile" in auth_url
        assert "state=" in auth_url
        assert len(state) == 43
        assert state in oauth_provider._state_store
 
    def test_generate_authorization_url_with_state(self, oauth_provider):
        custom_state = "custom_state_123"
        _, state = oauth_provider.generate_authorization_url(state=custom_state)
        
        assert state == custom_state
        assert custom_state in oauth_provider._state_store

    def test_verify_state_valid(self, oauth_provider):
        _, state = oauth_provider.generate_authorization_url()
        
        result = oauth_provider.verify_state(state)
        
        assert result is True
        assert state not in oauth_provider._state_store

    def test_verify_state_invalid_not_found(self, oauth_provider):
        result = oauth_provider.verify_state("nonexistent_state")
        
        assert result is False

    def test_verify_state_expired(self, oauth_provider):
        _, state = oauth_provider.generate_authorization_url()
        
        oauth_provider._state_store[state] = datetime.utcnow() - timedelta(minutes=10)
        
        result = oauth_provider.verify_state(state)
        
        assert result is False
        assert state not in oauth_provider._state_store

    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self, oauth_provider):
        with patch("app.mcp.oauth.oauth.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "new_access_token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": "new_refresh_token",
                "scope": "openid email"
            }
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            token = await oauth_provider.exchange_code_for_token("authorization_code")
            
            assert token.access_token == "new_access_token"
            assert token.token_type == "Bearer"
            assert token.expires_in == 3600
            assert token.refresh_token == "new_refresh_token"
            assert token.scope == "openid email"
    
    @pytest.mark.asyncio
    async def test_refresh_access_token(self, oauth_provider):
        with patch("app.mcp.oauth.oauth.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "refreshed_access_token",
                "token_type": "Bearer",
                "expires_in": 7200,
                "refresh_token": "new_refresh_token",
            }
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            token = await oauth_provider.refresh_access_token("old_refresh_token")
            
            assert token.access_token == "refreshed_access_token"
            assert token.expires_in == 7200
    
    @pytest.mark.asyncio
    async def test_get_user_info(self, oauth_provider):
        with patch("app.mcp.oauth.oauth.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {
                "sub": "1234567890",
                "name": "Test User",
                "email": "test@example.com"
            }
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            user_info = await oauth_provider.get_user_info("test_access_token")
            
            assert user_info["sub"] == "1234567890"
            assert user_info["name"] == "Test User"
            assert user_info["email"] == "test@example.com"


class TestGitHubOAuthProvider:

    def test_github_oauth_provider_initialization(self):
        with patch.dict("os.environ", {
            "GITHUB_CLIENT_ID": "test_github_client_id",
            "GITHUB_CLIENT_SECRET": "test_github_client_secret"
        }):
            provider = GitHubOAuthProvider(redirect_uri="http://localhost:8080/oauth/callback")
            
            assert provider.config.client_id == "test_github_client_id"
            assert provider.config.client_secret == "test_github_client_secret"
            assert provider.config.authorization_url == "https://github.com/login/oauth/authorize"
            assert provider.config.token_url == "https://github.com/login/oauth/access_token"
            assert provider.config.userinfo_url == "https://api.github.com/user"

    @pytest.mark.asyncio
    async def test_get_user_info_with_email(self, oauth_token):
        with patch("app.mcp.oauth.oauth.httpx.AsyncClient") as mock_client:
            mock_user_response = Mock()
            mock_user_response.json.return_value = {
                "login": "testuser",
                "name": "Test User",
                "email": "test@example.com"
            }
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_user_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            provider = GitHubOAuthProvider(redirect_uri="http://localhost:8080/oauth/callback")
            user_info = await provider.get_user_info("test_access_token")
            
            assert user_info["email"] == "test@example.com"


class TestGoogleOAuthProvider:

    def test_google_oauth_provider_initialization(self):
        """Test creating a GoogleOAuthProvider"""
        with patch.dict("os.environ", {
            "GOOGLE_CLIENT_ID": "test_google_client_id",
            "GOOGLE_CLIENT_SECRET": "test_google_client_secret"
        }):
            provider = GoogleOAuthProvider(redirect_uri="http://localhost:8080/oauth/callback")
            
            assert provider.config.client_id == "test_google_client_id"
            assert provider.config.client_secret == "test_google_client_secret"
            assert provider.config.authorization_url == "https://accounts.google.com/o/oauth2/v2/auth"
            assert provider.config.token_url == "https://oauth2.googleapis.com/token"
            assert provider.config.userinfo_url == "https://www.googleapis.com/oauth2/v2/userinfo"

    def test_google_oauth_provider_missing_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            # Ensure env vars are not set
            os.environ.pop("GOOGLE_CLIENT_ID", None)
            os.environ.pop("GOOGLE_CLIENT_SECRET", None)
            
            provider = GoogleOAuthProvider(redirect_uri="http://localhost:8080/oauth/callback")
            
            assert provider.config.client_id == ""
            assert provider.config.client_secret == ""


class TestOAuthManager:

    def test_get_provider_github(self, oauth_manager):
        provider = oauth_manager.get_provider("github")

        assert provider is not None
        assert isinstance(provider, GitHubOAuthProvider)

    def test_get_provider_google(self, oauth_manager):
        provider = oauth_manager.get_provider("google")
        
        assert provider is not None
        assert isinstance(provider, GoogleOAuthProvider)

    def test_get_provider_unknown(self, oauth_manager):
        provider = oauth_manager.get_provider("unknown")
        
        assert provider is None

    def test_store_and_get_token(self, oauth_manager, oauth_token):
        session_id = "test_session_123"
        
        oauth_manager.store_token(session_id, oauth_token)
        
        retrieved_token = oauth_manager.get_token(session_id)
        assert retrieved_token is not None
        assert retrieved_token.access_token == "test_access_token"

    def test_get_token_nonexistent(self, oauth_manager):
        token = oauth_manager.get_token("nonexistent_session")
        
        assert token is None

    def test_remove_token(self, oauth_manager, oauth_token):
        session_id = "test_session_123"
        
        oauth_manager.store_token(session_id, oauth_token)
        oauth_manager.remove_token(session_id)
        
        token = oauth_manager.get_token(session_id)
        assert token is None

    @pytest.mark.asyncio
    async def test_get_valid_token(self, oauth_manager, oauth_token):
        session_id = "test_session_123"
        oauth_manager.store_token(session_id, oauth_token)
        
        token = await oauth_manager.get_valid_token(session_id)
        
        assert token is not None
        assert token.access_token == "test_access_token"

    @pytest.mark.asyncio
    async def test_get_valid_token_expired_no_refresh(self, oauth_manager):
        expired_token = OAuthToken(
            access_token="expired_token",
            token_type="Bearer",
            expires_in=1,
            refresh_token=None
        )
        
        object.__setattr__(expired_token, "created_at", datetime.utcnow() - timedelta(hours=2))
        
        session_id = "test_session_123"
        oauth_manager.store_token(session_id, expired_token)
        
        token = await oauth_manager.get_valid_token(session_id)
        
        assert token is None

    @pytest.mark.asyncio
    async def test_get_valid_token_expired_with_refresh(self, oauth_manager, oauth_token):
        expired_token = OAuthToken(
            access_token="expired_token",
            token_type="Bearer",
            expires_in=1,
            refresh_token="refresh_token"
        )
        object.__setattr__(expired_token, "created_at", datetime.utcnow() - timedelta(hours=2))
        
        session_id = "test_session_123"
        oauth_manager.store_token(session_id, expired_token)
        
        for provider in oauth_manager.providers.values():
            provider.refresh_access_token = AsyncMock(return_value=oauth_token)
        
        token = await oauth_manager.get_valid_token(session_id)
        
        assert token is not None
        assert token.access_token == "test_access_token"

    def test_create_oauth_routes(self, oauth_manager):
        routes = oauth_manager.create_oauth_routes()
        
        assert len(routes) == 5
        route_paths = [route.path for route in routes]
        assert "/oauth/login" in route_paths
        assert "/oauth/authorize/{provider}" in route_paths
        assert "/oauth/callback/{provider}" in route_paths
        assert "/oauth/status" in route_paths
        assert "/oauth/logout" in route_paths
