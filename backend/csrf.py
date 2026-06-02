"""
backend/csrf.py — Signed Double Submit Cookie CSRF protection.
"""

import os
import hmac
import hashlib
import secrets
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# ── Constants & Configuration ──────────────────────────────────────────────────

CSRF_HEADER_NAME = "x-csrf-token"          # HTTP headers are lowercased by Starlette
CSRF_TOKEN_BYTES = 32                       # 256-bit token → 64 hex chars
CSRF_COOKIE_MAX_AGE = 60 * 60 * 8          # 8 hours in seconds

# Enforced secure variant blocks subdomain manipulations
CSRF_COOKIE_NAME = "__Host-csrftoken"  

# Loaded securely from configuration context
CSRF_SECRET_KEY = os.environ.get("CSRF_SECRET_KEY", "dev-fallback-key").encode("utf-8")

# Parse allowed origins comma-separated string from environments
_ALLOWED_ORIGINS_RAW = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
ALLOWED_ORIGINS = {origin.strip() for origin in _ALLOWED_ORIGINS_RAW.split(",") if origin.strip()}


# ── Response schema ───────────────────────────────────────────────────────────

class CSRFTokenResponse(BaseModel):
    """OpenAPI response schema for GET /api/csrf-token."""
    csrfToken: str


_PROTECTED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_EXEMPT_PATHS: set[str] = {"/api/feedback"}


# ── Token helpers ─────────────────────────────────────────────────────────────

def generate_csrf_token() -> str:
    """Return a new cryptographically secure hex token."""
    return secrets.token_hex(CSRF_TOKEN_BYTES)


def _is_secure_context() -> bool:
    """Return True when the app is running behind HTTPS."""
    if os.environ.get("TESTING", "").strip().lower() in ("true", "1", "yes"):
        return False
    val = os.environ.get("CSRF_SECURE", "true").strip().lower()
    return val not in ("false", "0", "no")


def set_csrf_cookie(response: Response, token: str) -> None:
    """Write the SIGNED CSRF token into a cookie and set cache headers."""
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    # Cryptographically sign the token payload
    signature = hmac.new(CSRF_SECRET_KEY, token.encode("utf-8"), hashlib.sha256).hexdigest()
    signed_value = f"{token}.{signature}"

    secure_ctx = _is_secure_context()
    cookie_name = CSRF_COOKIE_NAME if secure_ctx else "csrftoken" 

    response.set_cookie(
        key=cookie_name,
        value=signed_value,
        max_age=CSRF_COOKIE_MAX_AGE,
        path="/",
        samesite="lax",
        httponly=False,
        secure=secure_ctx,
    )


# ── Middleware ────────────────────────────────────────────────────────────────

class CSRFMiddleware:
    """Pure ASGI CSRF middleware using Signed Double Submit Cookies."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)

        # 1. Skip safe methods
        if request.method.upper() not in _PROTECTED_METHODS:
            await self._app(scope, receive, send)
            return

        # 2. Skip explicitly exempt paths
        if request.url.path in _EXEMPT_PATHS:
            await self._app(scope, receive, send)
            return

        # 2.5 Strict Origin/Referer validation for secure contexts (OWASP defense-in-depth)
        if _is_secure_context():
            # Extract the source domain via Origin header, fallback to Referer if absent
            source_origin = request.headers.get("origin")
            if not source_origin and request.headers.get("referer"):
                # Basic parsing to extract origin protocol + host from full referer string
                from starlette.datastructures import URL as StarletteURL
                ref = request.headers.get("referer", "")
                try:
                    parsed_ref = StarletteURL(ref)
                    source_origin = f"{parsed_ref.scheme}://{parsed_ref.netloc}"
                except Exception:
                    source_origin = None

            if not source_origin or source_origin not in ALLOWED_ORIGINS:
                logger.warning(
                    "CSRF validation failed (Origin mismatch/unauthorized source: %s) path=%s method=%s",
                    source_origin, request.url.path, request.method
                )
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed: origin unauthorized."},
                )
                await response(scope, receive, send)
                return

       # 3. Read raw data values securely (Strict enforcement based on context)
        secure_ctx = _is_secure_context()
        expected_cookie_name = CSRF_COOKIE_NAME if secure_ctx else "csrftoken"

        cookie_value = request.cookies.get(expected_cookie_name, "")
        header_token = request.headers.get(CSRF_HEADER_NAME, "")

        # 4. Reject immediately if either element is absent entirely
        if not cookie_value or not header_token:
            logger.warning(
                "CSRF validation failed (missing tokens) path=%s method=%s",
                request.url.path, request.method
            )
            response = JSONResponse(
                status_code=403,
                content={"detail": "CSRF token missing."},
            )
            await response(scope, receive, send)
            return

        # 5. Extract and Validate Cryptographic Structure
        if "." not in cookie_value:
            logger.warning("CSRF validation failed (malformed signature structure) path=%s", request.url.path)
            response = JSONResponse(
                status_code=403,
                content={"detail": "CSRF token invalid."},
            )
            await response(scope, receive, send)
            return

        raw_token_from_cookie, signature = cookie_value.split(".", 1)
        
        # Recalculate expected validation signature
        expected_sig = hmac.new(CSRF_SECRET_KEY, raw_token_from_cookie.encode("utf-8"), hashlib.sha256).hexdigest()

        # 6. Verify signature validity AND match against the header in constant-time
        is_signature_valid = hmac.compare_digest(signature, expected_sig)
        is_header_valid = hmac.compare_digest(raw_token_from_cookie, header_token)

        if not (is_signature_valid and is_header_valid):
            logger.warning(
                "CSRF validation failed (signature match: %s, token match: %s) path=%s method=%s",
                is_signature_valid, is_header_valid, request.url.path, request.method
            )
            response = JSONResponse(
                status_code=403,
                content={"detail": "CSRF token invalid."},
            )
            await response(scope, receive, send)
            return

        # 7. Complete Validation Passed
        await self._app(scope, receive, send)