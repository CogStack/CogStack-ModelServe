import os
import secrets
from typing import Optional
from fastapi import Request, Response, HTTPException, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import FastAPI
from app.mcp.oauth.oauth import OAuthManager
from app.mcp.logger import get_logger

logger = get_logger(__name__)


def register_oauth_routes(app: FastAPI, oauth_manager: OAuthManager) -> None:

    @app.get("/oauth/login", response_class=HTMLResponse)
    async def oauth_login() -> HTMLResponse:
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
                .btn-oauth svg {
                    margin-right: 10px;
                    width: 20px;
                    height: 20px;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <h1>🔐 CMS MCP Server</h1>
                <p>Sign in to access the MCP server</p>

                <a href="/oauth/authorize/google" class="btn-oauth btn-google">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                    </svg>
                    Continue with Google
                </a>

                <a href="/oauth/authorize/github" class="btn-oauth btn-github">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                    </svg>
                    Continue with GitHub
                </a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.get("/oauth/authorize/{provider}")
    async def oauth_authorize(provider: str, response: Response) -> RedirectResponse:
        oauth_provider = oauth_manager.get_provider(provider)
        if not oauth_provider:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

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

    @app.get("/oauth/callback/{provider}")
    async def oauth_callback(
            request: Request,
            provider: str,
            code: Optional[str] = None,
            state: Optional[str] = None,
            error: Optional[str] = None,
    ) -> HTMLResponse:
        if error:
            logger.error(f"OAuth error for {provider}: {error}")
            return HTMLResponse(
                content=f"""
                <!DOCTYPE html>
                <html>
                <head><title>Authentication Error</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>❌ Authentication Failed</h1>
                    <p>Error: {error}</p>
                    <a href="/oauth/login" style="color: #667eea;">Try again</a>
                </body>
                </html>
                """,
                status_code=400
            )

        if not code or not state:
            raise HTTPException(status_code=400, detail="Missing code or state")

        if request is None:
            raise HTTPException(status_code=400, detail="Request object is required")

        oauth_provider = oauth_manager.get_provider(provider)
        if not oauth_provider:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

        stored_state = request.cookies.get(f"oauth_state_{provider}")
        if not stored_state or stored_state != state:
            logger.error(f"State mismatch for {provider}")
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        if not oauth_provider.verify_state(state):
            logger.error(f"State verification failed for {provider}")
            raise HTTPException(status_code=400, detail="State verification failed")

        try:
            token = await oauth_provider.exchange_code_for_token(code)
            user_info = await oauth_provider.get_user_info(token.access_token)
            session_id = secrets.token_urlsafe(32)
            oauth_manager.store_token(session_id, token)
            user_email = user_info.get("email", "N/A")
            user_name = user_info.get("name") or user_info.get("login", "User")

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Successful</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
                    .success-icon {{
                        font-size: 64px;
                        margin-bottom: 1rem;
                    }}
                    h1 {{
                        color: #333;
                        margin-bottom: 1rem;
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
                    <div class="success-icon">✅</div>
                    <h1>Authentication Successful!</h1>
                    <div class="user-info">
                        <p><strong>Welcome, {user_name}!</strong></p>
                        <p>Email: {user_email}</p>
                        <p>Provider: {provider.title()}</p>
                    </div>
                    <div class="token-info">
                        <p><strong>Session ID:</strong></p>
                        <p>{session_id}</p>
                        <p style="margin-top: 10px; font-size: 11px; color: #666;">
                            Use this session ID in your MCP client configuration or include the session cookie.
                        </p>
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
            raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

    @app.get("/oauth/status")
    async def oauth_status(
            request: Request,
            cms_mcp_session: Optional[str] = Cookie(None)
    ) -> HTMLResponse:
        if not cms_mcp_session:
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                <head><title>No Session</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>🔒 No Active Session</h1>
                    <p>You are not currently authenticated.</p>
                    <a href="/oauth/login" style="color: #667eea;">Login</a>
                </body>
                </html>
                """
            )

        token = await oauth_manager.get_valid_token(cms_mcp_session)

        if not token:
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                <head><title>Session Expired</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>⏰ Session Expired</h1>
                    <p>Your session has expired. Please login again.</p>
                    <a href="/oauth/login" style="color: #667eea;">Login</a>
                </body>
                </html>
                """
            )

        is_expired = "Yes" if token.is_expired() else "No"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Session Status</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
                    margin-bottom: 1rem;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 0.5rem 0;
                    border-bottom: 1px solid #eee;
                }}
                .btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: #dc3545;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="status-container">
                <h1>✅ Active Session</h1>
                <div class="info-row">
                    <strong>Session ID:</strong>
                    <span>{cms_mcp_session[:16]}...</span>
                </div>
                <div class="info-row">
                    <strong>Token Type:</strong>
                    <span>{token.token_type}</span>
                </div>
                <div class="info-row">
                    <strong>Expires In:</strong>
                    <span>{token.expires_in} seconds</span>
                </div>
                <div class="info-row">
                    <strong>Is Expired:</strong>
                    <span>{is_expired}</span>
                </div>
                <div class="info-row">
                    <strong>Has Refresh Token:</strong>
                    <span>{"Yes" if token.refresh_token else "No"}</span>
                </div>
                <a href="/oauth/logout" class="btn">Logout</a>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    @app.get("/oauth/logout")
    async def oauth_logout(cms_mcp_session: Optional[str] = Cookie(None)) -> RedirectResponse:
        if cms_mcp_session:
            oauth_manager.remove_token(cms_mcp_session)

        response = RedirectResponse(url="/oauth/login")
        response.delete_cookie("cms_mcp_session")

        logger.info("User logged out")
        return response
