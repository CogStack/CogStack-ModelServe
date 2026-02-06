from typing import Optional, Callable, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from app.mcp.oauth.oauth import OAuthManager
from app.mcp.logger import get_logger

logger = get_logger(__name__)


class OAuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: Any, oauth_manager: OAuthManager, public_paths: Optional[list[str]] = None):
        super().__init__(app)
        self.oauth_manager = oauth_manager
        self.public_paths = public_paths or []

    def _is_public_path(self, path: str) -> bool:
        return any(path.startswith(public_path) for public_path in self.public_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            if request.url.path.startswith("/sse"):
                return await call_next(request)

            if self._is_public_path(request.url.path):
                return await call_next(request)

            session_id = request.cookies.get("cms_mcp_session")

            if not session_id:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    session_id = auth_header[7:]

            if not session_id:
                session_id = request.query_params.get("session_id")

            if not session_id:
                logger.warning(f"No session provided for path: {request.url.path}")
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "message": "No session token provided. Please authenticate at /oauth/login"
                    }
                )

            logger.debug(f"Validating session: {session_id[:16]}...")
            token = await self.oauth_manager.get_valid_token(session_id)

            if not token:
                logger.warning(f"Invalid or expired session: {session_id[:16]}...")
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "message": "Invalid or expired session. Please re-authenticate at /oauth/login"
                    }
                )

            request.state.oauth_token = token
            request.state.session_id = session_id

            response = await call_next(request)

            return response
        except Exception as e:
            logger.error(f"Unhandled exception in OAuth middleware: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": f"An unexpected error occurred: {str(e)}"
                }
            )


def get_oauth_token_from_request(request: Request) -> Optional[str]:
    if hasattr(request.state, "oauth_token"):
        return request.state.oauth_token.access_token
    return None


def require_auth(request: Request) -> str:
    token = get_oauth_token_from_request(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return token
