import os
import secrets
import httpx
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request
from starlette.routing import Route
from starlette.responses import Response
from app.mcp.logger import get_logger


logger = get_logger(__name__)


@dataclass
class OAuthConfig:
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    userinfo_url: str
    redirect_uri: str
    scope: str


@dataclass
class OAuthToken:
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def is_expired(self) -> bool:
        if not self.expires_in:
            return False
        created = self.created_at
        assert created is not None
        expiry_time = created + timedelta(seconds=self.expires_in)
        return datetime.utcnow() >= expiry_time


class OAuthProvider:

    def __init__(self, config: OAuthConfig):
        self.config = config
        self._state_store: Dict[str, datetime] = {}  # Use Redis or database for production
        self._token_store: Dict[str, OAuthToken] = {}  # Use secure storage for production

    def generate_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        if state is None:
            state = secrets.token_urlsafe(32)

        self._state_store[state] = datetime.utcnow()

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": self.config.scope,
            "state": state,
        }

        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        auth_url = f"{self.config.authorization_url}?{query_string}"

        return auth_url, state

    def verify_state(self, state: str) -> bool:
        if state not in self._state_store:
            return False

        created_at = self._state_store[state]
        if datetime.utcnow() - created_at > timedelta(minutes=5):
            del self._state_store[state]
            return False

        del self._state_store[state]
        return True

    async def exchange_code_for_token(self, code: str) -> OAuthToken:
        async with httpx.AsyncClient() as client:
            data = {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "grant_type": "authorization_code",
            }

            response = await client.post(
                self.config.token_url,
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            token_data = response.json()

            token = OAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in", 3600),
                refresh_token=token_data.get("refresh_token"),
                scope=token_data.get("scope"),
            )

            logger.info("Successfully exchanged code for access token")
            return token

    async def refresh_access_token(self, refresh_token: str) -> OAuthToken:
        async with httpx.AsyncClient() as client:
            data = {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }

            response = await client.post(
                self.config.token_url,
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            token_data = response.json()

            token = OAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in", 3600),
                refresh_token=token_data.get("refresh_token", refresh_token),
                scope=token_data.get("scope"),
            )

            logger.info("Successfully refreshed access token")
            return token

    async def get_user_info(self, access_token: str) -> dict:
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }

            response = await client.get(
                self.config.userinfo_url,
                headers=headers
            )
            response.raise_for_status()
            return response.json()


class GoogleOAuthProvider(OAuthProvider):

    def __init__(self, redirect_uri: str):
        config = OAuthConfig(
            client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
            authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
            redirect_uri=redirect_uri,
            scope="openid email profile",
        )
        super().__init__(config)

        if not config.client_id or not config.client_secret:
            logger.warning("Google OAuth credentials not configured")


class GitHubOAuthProvider(OAuthProvider):

    def __init__(self, redirect_uri: str):
        config = OAuthConfig(
            client_id=os.getenv("GITHUB_CLIENT_ID", ""),
            client_secret=os.getenv("GITHUB_CLIENT_SECRET", ""),
            authorization_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            redirect_uri=redirect_uri,
            scope="read:user user:email",
        )
        super().__init__(config)

        if not config.client_id or not config.client_secret:
            logger.warning("GitHub OAuth credentials not configured")

    async def get_user_info(self, access_token: str) -> dict:
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }

            user_response = await client.get(self.config.userinfo_url, headers=headers)
            user_response.raise_for_status()
            user_data = user_response.json()

            if not user_data.get("email"):
                logger.debug("User email not in primary user info, fetching from /user/emails")
                emails_response = await client.get("https://api.github.com/user/emails", headers=headers)
                if emails_response.status_code == 200:
                    emails_data = emails_response.json()
                    for email_info in emails_data:
                        if email_info.get("primary"):
                            user_data["email"] = email_info.get("email")
                            break
                    if not user_data.get("email"):
                        for email_info in emails_data:
                            if email_info.get("verified"):
                                user_data["email"] = email_info.get("email")
                                break
                else:
                    logger.warning(f"Failed to fetch user emails: {emails_response.status_code}")

            return user_data


class OAuthManager:

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.providers = {
            "google": GoogleOAuthProvider(f"{base_url}/oauth/callback/google"),
            "github": GitHubOAuthProvider(f"{base_url}/oauth/callback/github"),
        }
        self._sessions: Dict[str, OAuthToken] = {}

    def get_provider(self, provider_name: str) -> Optional[OAuthProvider]:
        return self.providers.get(provider_name.lower())

    def store_token(self, session_id: str, token: OAuthToken) -> None:
        self._sessions[session_id] = token
        logger.info(f"Stored token for session: {session_id}")

    def get_token(self, session_id: str) -> Optional[OAuthToken]:
        return self._sessions.get(session_id)

    def remove_token(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Removed token for session: {session_id}")

    async def get_valid_token(self, session_id: str) -> Optional[OAuthToken]:
        token = self.get_token(session_id)
        if not token:
            return None

        if token.is_expired() and token.refresh_token:
            for provider in self.providers.values():
                try:
                    new_token = await provider.refresh_access_token(token.refresh_token)
                    self.store_token(session_id, new_token)
                    return new_token
                except Exception as e:
                    logger.debug(f"Failed to refresh token with provider: {e}")
                    continue

            self.remove_token(session_id)
            return None

        return token if not token.is_expired() else None

    def create_oauth_routes(self) -> list:
        async def oauth_login(request: Request) -> Response:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>CMS MCP Server - Login</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }
                    .login-container {
                        background: white;
                        padding: 2rem;
                        border-radius: 10px;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 400px;
                        width: 90%;
                        box-sizing: border-box;
                    }
                    h1 {
                        color: #333;
                        margin-bottom: 0.5rem;
                    }
                    p {
                        color: #666;
                        margin-bottom: 2rem;
                    }
                    .btn-oauth {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        width: 100%;
                        padding: 12px 20px;
                        margin: 10px 0;
                        border: none;
                        border-radius: 5px;
                        font-size: 16px;
                        font-weight: 500;
                        cursor: pointer;
                        text-decoration: none;
                        transition: transform 0.2s, box-shadow 0.2s;
                        box-sizing: border-box;
                    }
                    .btn-oauth:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    }
                    .btn-google {
                        background: #4285f4;
                        color: white;
                    }
                    .btn-github {
                        background: #24292e;
                        color: white;
                    }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <h1>🔐 CMS MCP Server</h1>
                    <p>Sign in to access the MCP server</p>

                    <a href="/oauth/authorize/google" class="btn-oauth btn-google">
                        Continue with Google
                    </a>

                    <a href="/oauth/authorize/github" class="btn-oauth btn-github">
                        Continue with GitHub
                    </a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)

        async def oauth_authorize(request: Request) -> Response:
            provider = request.path_params['provider']
            oauth_provider = self.get_provider(provider)

            if not oauth_provider:
                return HTMLResponse(content=f"<h1>Unknown provider: {provider}</h1>", status_code=400)

            auth_url, state = oauth_provider.generate_authorization_url()

            response = RedirectResponse(url=auth_url)
            response.set_cookie(
                key=f"oauth_state_{provider}",
                value=state,
                max_age=300,
                httponly=True,
                samesite="lax",
                secure=os.getenv("CMS_MCP_SECURE_COOKIES", "false").lower() == "true"
            )

            logger.info(f"Initiating OAuth flow for provider: {provider}")
            return response

        async def oauth_callback(request: Request) -> Response:
            provider = request.path_params['provider']
            code = request.query_params.get('code')
            state = request.query_params.get('state')
            error = request.query_params.get('error')

            if error:
                logger.error(f"OAuth error for {provider}: {error}")
                return HTMLResponse(
                    content=f"<h1>❌ Authentication Failed</h1><p>Error: {error}</p>",
                    status_code=400
                )

            if not code or not state:
                return HTMLResponse(content="<h1>Missing code or state</h1>", status_code=400)

            oauth_provider = self.get_provider(provider)
            if not oauth_provider:
                return HTMLResponse(content=f"<h1>Unknown provider: {provider}</h1>", status_code=400)

            stored_state = request.cookies.get(f"oauth_state_{provider}")
            if not stored_state or stored_state != state:
                logger.error(f"State mismatch for {provider}")
                return HTMLResponse(content="<h1>Invalid state parameter</h1>", status_code=400)

            if not oauth_provider.verify_state(state):
                logger.error(f"State verification failed for {provider}")
                return HTMLResponse(content="<h1>State verification failed</h1>", status_code=400)

            try:
                token = await oauth_provider.exchange_code_for_token(code)
                user_info = await oauth_provider.get_user_info(token.access_token)
                session_id = secrets.token_urlsafe(32)

                self.store_token(session_id, token)

                user_email = user_info.get("email", "N/A")
                user_name = user_info.get("name") or user_info.get("login", "User")

                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Authentication Successful</title>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                            margin: 0;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        }}
                        .success-container {{
                            background: white;
                            padding: 2rem;
                            border-radius: 10px;
                            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                            text-align: center;
                            max-width: 500px;
                        }}
                        h1 {{
                            color: #333;
                        }}
                        .user-info {{
                            background: #f5f5f5;
                            padding: 1rem;
                            border-radius: 5px;
                            margin: 1rem 0;
                        }}
                        .token-info {{
                            background: #e8f5e9;
                            padding: 1rem;
                            border-radius: 5px;
                            margin: 1rem 0;
                            word-break: break-all;
                            font-family: monospace;
                            font-size: 12px;
                        }}
                        .btn {{
                            display: inline-block;
                            padding: 10px 20px;
                            background: #667eea;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                            margin-top: 1rem;
                        }}
                    </style>
                </head>
                <body>
                    <div class="success-container">
                        <div style="font-size: 64px;">✅</div>
                        <h1>Authentication Successful!</h1>
                        <div class="user-info">
                            <p><strong>Welcome, {user_name}!</strong></p>
                            <p>Email: {user_email}</p>
                            <p>Provider: {provider.title()}</p>
                        </div>
                        <div class="token-info">
                            <p><strong>Session ID:</strong></p>
                            <p>{session_id}</p>
                        </div>
                        <p style="color: #666; font-size: 14px;">
                            You can now use the MCP server with your authenticated session.
                        </p>
                        <a href="/oauth/status" class="btn">Check Session Status</a>
                    </div>
                </body>
                </html>
                """

                response = HTMLResponse(content=html_content)
                response.set_cookie(
                    key="cms_mcp_session",
                    value=session_id,
                    max_age=86400,
                    httponly=True,
                    samesite="lax",
                    secure=os.getenv("CMS_MCP_SECURE_COOKIES", "false").lower() == "true"
                )
                response.delete_cookie(f"oauth_state_{provider}")

                logger.info(f"Successfully authenticated user via {provider}: {user_email}")
                return response

            except Exception as e:
                logger.error(f"OAuth callback error for {provider}: {str(e)}")
                return HTMLResponse(
                    content=f"<h1>Authentication failed</h1><p>{str(e)}</p>",
                    status_code=500
                )

        async def oauth_status(request: Request) -> Response:
            session_id = request.cookies.get("cms_mcp_session")

            if not session_id:
                return HTMLResponse(content="<h1>🔒 No Active Session</h1><p>You are not authenticated.</p>")

            token = await self.get_valid_token(session_id)

            if not token:
                return HTMLResponse(content="<h1>⏰ Session Expired</h1><p>Please login again.</p>")

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Session Status</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }}
                    .status-container {{
                        background: white;
                        padding: 2rem;
                        border-radius: 10px;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                        max-width: 500px;
                    }}
                    h1 {{
                        color: #333;
                    }}
                    .info-row {{
                        display: flex;
                        justify-content: space-between;
                        padding: 0.5rem 0;
                        border-bottom: 1px solid #eee;
                    }}
                </style>
            </head>
            <body>
                <div class="status-container">
                    <h1>✅ Active Session</h1>
                    <div class="info-row">
                        <strong>Session ID:</strong>
                        <span>{session_id[:16]}...</span>
                    </div>
                    <div class="info-row">
                        <strong>Token Valid:</strong>
                        <span>Yes</span>
                    </div>
                    <div class="info-row">
                        <strong>Expires In:</strong>
                        <span>{token.expires_in} seconds</span>
                    </div>
                </div>
            </body>
            </html>
            """

            return HTMLResponse(content=html_content)

        async def oauth_logout(request: Request) -> Response:
            session_id = request.cookies.get("cms_mcp_session")
            if session_id:
                self.remove_token(session_id)

            response = RedirectResponse(url="/oauth/login")
            response.delete_cookie("cms_mcp_session")

            logger.info("User logged out")
            return response

        return [
            Route("/oauth/login", oauth_login),
            Route("/oauth/authorize/{provider}", oauth_authorize),
            Route("/oauth/callback/{provider}", oauth_callback),
            Route("/oauth/status", oauth_status),
            Route("/oauth/logout", oauth_logout),
        ]
