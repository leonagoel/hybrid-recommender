# backend/csrf.py
"""
CSRF protection middleware and utilities for the Hybrid Recommender API.
Implements the Double Submit Cookie pattern.
"""
import secrets
from fastapi import HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel


class CSRFTokenResponse(BaseModel):
    """Response model for CSRF token endpoint."""
    csrfToken: str


def generate_csrf_token() -> str:
    """Generate a cryptographically secure CSRF token."""
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response: Response, token: str) -> None:
    """Set CSRF token in an HTTP-only cookie."""
    response.set_cookie(
        "csrftoken",
        token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=3600,
    )
    response.headers["Cache-Control"] = "no-store"


class CSRFMiddleware:
    """
    CSRF protection middleware using the Double Submit Cookie pattern.
    
    Validates that:
    1. The token in the X-CSRF-Token header matches
    2. The token in the csrftoken cookie matches
    
    Safe methods (GET, HEAD, OPTIONS) skip validation.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        
        # Skip CSRF check for safe methods and health checks
        if method in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        
        # Skip for health check endpoints
        if path in ("/health", "/api/health"):
            await self.app(scope, receive, send)
            return
        
        # Extract headers
        headers = dict(scope.get("headers", []))
        csrf_token_header = headers.get(b"x-csrf-token", b"").decode("utf-8", errors="ignore").strip()
        
        # Extract cookies
        cookie_header = headers.get(b"cookie", b"").decode("utf-8", errors="ignore")
        csrf_token_cookie = ""
        for cookie in cookie_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith("csrftoken="):
                csrf_token_cookie = cookie.split("=", 1)[1].strip()
                break
        
        # Validate tokens
        if not csrf_token_header or not csrf_token_cookie or csrf_token_header != csrf_token_cookie:
            # Return 403 Forbidden
            await send({
                "type": "http.response.start",
                "status": 403,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"detail":"CSRF token missing or mismatched"}',
            })
            return
        
        await self.app(scope, receive, send)