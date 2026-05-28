"""
backend/csrf.py — Double Submit Cookie CSRF protection.

How it works:
  1. Client calls GET /api/csrf-token.
  2. Server generates a cryptographically secure token, writes it into a
     cookie named `csrftoken` (readable by JS — NOT HttpOnly), and also
     returns it in the JSON body so the frontend can store it in memory.
  3. On every state-mutating request (POST / PUT / PATCH / DELETE) the
     client must echo the same token value back in the `X-CSRF-Token`
     request header.
  4. CSRFMiddleware reads both values and compares them with a
     constant-time comparison.  A mismatch → 403.

Why Double Submit Cookie is correct here:
  - The project has no server-side sessions, so the Synchronizer Token
    pattern (server stores token per session) is not applicable.
  - The backend does not use cookies for authentication, so an attacker
    cannot exploit cookie-based auth via CSRF.
  - A cross-origin page cannot read the `csrftoken` cookie value
    (SameSite + CORS restrictions), so it cannot forge the header.

Skipped methods: GET, HEAD, OPTIONS  (safe / pre-flight).
WebSocket upgrades are also skipped — they are not affected by CSRF.
"""

import os
import secrets
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CSRF_COOKIE_NAME = "csrftoken"
CSRF_HEADER_NAME = "x-csrf-token"          # HTTP headers are lowercased by Starlette
CSRF_TOKEN_BYTES = 32                       # 256-bit token → 64 hex chars
CSRF_COOKIE_MAX_AGE = 60 * 60 * 8          # 8 hours in seconds


# ── Response schema ───────────────────────────────────────────────────────────

class CSRFTokenResponse(BaseModel):
    """
    OpenAPI response schema for GET /api/csrf-token.

    Documenting the response shape in Pydantic means:
      - FastAPI generates accurate OpenAPI/Swagger docs for this endpoint.
      - Clients using auto-generated SDK code get a typed model.
      - The field name `csrfToken` is explicit and validated on output.
    """
    csrfToken: str  # 64-character hex string (256-bit entropy)

# Methods that mutate state and therefore require a valid CSRF token.
_PROTECTED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Paths that are explicitly exempt from CSRF validation even when they use a
# protected method.  Add paths here only when you have a strong reason (e.g.
# a webhook endpoint that is validated by a shared secret instead).
_EXEMPT_PATHS: set[str] = {"/api/feedback"}


# ── Token helpers ─────────────────────────────────────────────────────────────

def generate_csrf_token() -> str:
    """Return a new cryptographically secure hex token."""
    return secrets.token_hex(CSRF_TOKEN_BYTES)


def _is_secure_context() -> bool:
    """
    Return True when the app is running behind HTTPS.
    Controlled by the CSRF_SECURE env var.

    Production default: True  (cookie is HTTPS-only).
    Local HTTP dev:     set CSRF_SECURE=false in your .env.
    Test runner:        automatically False when TESTING=true,
                        so TestClient (plain HTTP) can send the cookie.
    """
    # Allow the test harness to force insecure mode without requiring
    # every developer to remember to set CSRF_SECURE=false locally.
    if os.environ.get("TESTING", "").strip().lower() in ("true", "1", "yes"):
        return False
    val = os.environ.get("CSRF_SECURE", "true").strip().lower()
    return val not in ("false", "0", "no")


def set_csrf_cookie(response: Response, token: str) -> None:
    """
    Write the CSRF token into a cookie and set cache-prevention headers.

    Cookie flags — every flag is a deliberate security decision:

      httponly=False
        The Double Submit Cookie pattern REQUIRES JavaScript to read this
        cookie value so it can copy it into the X-CSRF-Token request header.
        Setting HttpOnly=True would break the pattern entirely because JS
        cannot read HttpOnly cookies.  The security model relies on the
        Same-Origin Policy: a cross-origin attacker cannot read this cookie
        even though it is not HttpOnly.

      samesite="lax"
        Blocks the cookie from being sent on cross-site POST/PUT/PATCH/DELETE
        requests initiated by third-party pages (form submissions, AJAX from
        other origins).  This is the primary browser-level CSRF defence.
        "Lax" still allows the cookie on top-level GET navigations, which is
        required for OAuth redirect flows.  Use "strict" only if you are
        certain no cross-site GET navigation needs to carry the cookie.

      secure=_is_secure_context()
        When True (production), the browser only sends this cookie over HTTPS.
        This prevents the token from being transmitted in plaintext over HTTP,
        which would allow a network attacker to steal it.
        Disabled via CSRF_SECURE=false for local HTTP development only.

      path="/"
        The cookie is sent on every request to this origin, not just a
        specific path prefix.  Required because API endpoints are spread
        across /api/* and the frontend is served from /.

      max_age=CSRF_COOKIE_MAX_AGE
        8-hour expiry.  Uses max_age (seconds, relative) rather than expires
        (absolute datetime) to avoid clock-skew issues between client and
        server.  The browser deletes the cookie automatically after expiry.

    Response headers:

      Cache-Control: no-store
        Prevents any proxy, CDN, or browser cache from storing the token
        response.  A cached CSRF token response would allow an attacker to
        replay a stale token or observe a token issued to another user.

      Pragma: no-cache
        HTTP/1.0 cache-control compatibility header.  Modern proxies use
        Cache-Control, but Pragma ensures older intermediaries also do not
        cache this response.
    """
    # Prevent any cache layer from storing the token response.
    # This must be set on the Response object before the route handler returns.
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    response.set_cookie(
        key=CSRF_COOKIE_NAME,       # Cookie name the browser stores and sends
        value=token,                # Raw 64-char hex token — no encoding needed
        max_age=CSRF_COOKIE_MAX_AGE,# Relative expiry in seconds (8 hours)
        path="/",                   # Sent on all routes, not just /api/*
        samesite="lax",             # Blocks cross-site POST; allows OAuth GETs
        httponly=False,             # JS must read this — see docstring above
        secure=_is_secure_context(),# HTTPS-only in production
    )


# ── Middleware ────────────────────────────────────────────────────────────────

class CSRFMiddleware:
    """
    Pure ASGI CSRF middleware using the Double Submit Cookie pattern.

    Implemented as a raw ASGI callable instead of BaseHTTPMiddleware.

    WHY NOT BaseHTTPMiddleware?
    BaseHTTPMiddleware buffers the entire response body before sending it.
    When this middleware short-circuits with a 403 JSONResponse, the outer
    response_time_middleware (registered via @app.middleware("http")) tries
    to append X-Response-Time to the already-consumed response object,
    raising a MutableHeaders error on Starlette >= 0.20.  The raw ASGI
    approach avoids all buffering — the 403 is sent directly down the
    ASGI send channel without touching any outer middleware's response object.

    Registration in main.py:
        app.add_middleware(CSRFMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        # Store the next ASGI app in the middleware chain.
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only intercept HTTP requests — pass WebSocket and lifespan through.
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)

        # 1. Skip safe methods (GET, HEAD, OPTIONS) — they never mutate state.
        if request.method.upper() not in _PROTECTED_METHODS:
            await self._app(scope, receive, send)
            return

        # 2. Skip explicitly exempt paths (e.g. HMAC-signed webhooks).
        if request.url.path in _EXEMPT_PATHS:
            await self._app(scope, receive, send)
            return

        # 3. Read the token from the inbound cookie header.
        #    request.cookies is a plain dict populated by Starlette from
        #    the raw Cookie header — no browser involvement needed.
        cookie_token: str = request.cookies.get(CSRF_COOKIE_NAME, "")

        # 4. Read the token from the custom request header.
        #    Starlette stores all header names in lowercase, so we compare
        #    against the lowercase constant CSRF_HEADER_NAME = "x-csrf-token".
        header_token: str = request.headers.get(CSRF_HEADER_NAME, "")

        # 5. Reject immediately if either value is absent or empty.
        if not cookie_token or not header_token:
            logger.warning(
                "CSRF validation failed (missing token) path=%s method=%s "
                "cookie_present=%s header_present=%s",
                request.url.path,
                request.method,
                bool(cookie_token),
                bool(header_token),
            )
            # Build the 403 response and send it directly through the ASGI
            # send channel — bypasses all outer middleware response objects.
            response = JSONResponse(
                status_code=403,
                content={"detail": "CSRF token missing."},
            )
            await response(scope, receive, send)
            return

        # 6. Constant-time comparison prevents timing side-channel attacks.
        #    A regular == short-circuits on the first differing byte, leaking
        #    information about how many leading characters match.
        if not secrets.compare_digest(cookie_token, header_token):
            logger.warning(
                "CSRF validation failed (token mismatch) path=%s method=%s",
                request.url.path,
                request.method,
            )
            response = JSONResponse(
                status_code=403,
                content={"detail": "CSRF token invalid."},
            )
            await response(scope, receive, send)
            return

        # 7. Both tokens present and identical — pass to the next handler.
        await self._app(scope, receive, send)
