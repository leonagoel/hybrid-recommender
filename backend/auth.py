"""Authentication dependencies for the FastAPI backend.

Provides:
    - Admin API key verification for admin-only endpoints.
    - Supabase JWT verification (signature + expiry + audience) for
      user-scoped endpoints.
"""

import os

import jwt
from fastapi import Header, HTTPException, Request

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")

SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_JWT_AUDIENCE = os.environ.get(
    "SUPABASE_JWT_AUDIENCE", "authenticated"
).strip()


def _extract_bearer_token(value: str | None) -> str:
    """Pull the raw token out of an Authorization: Bearer <token> header."""
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def verify_supabase_jwt(token: str) -> dict:
    """Verify a Supabase-issued access token.

    Checks, in order:
        1. The token is signed with our configured secret (HS256).
        2. The `exp` claim has not passed.
        3. The `aud` claim matches the expected audience.

    Returns the decoded claims dict on success.
    Raises HTTPException(401) on any verification failure.
    """
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_JWT_SECRET is not configured on the server.",
        )

    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    try:
        claims = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience=SUPABASE_JWT_AUDIENCE,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=401, detail="Token audience is invalid."
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication token."
        )

    return claims


def get_current_user_id(request: Request) -> str:
    """FastAPI dependency: verify the caller's JWT, return their user id."""
    token = _extract_bearer_token(request.headers.get("authorization"))
    claims = verify_supabase_jwt(token)

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401, detail="Token is missing a subject claim."
        )

    return user_id


def _require_admin_access(x_admin_key: str = Header(default=None)):
    """Require a valid admin API key for protected endpoints."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY is not configured on the server.",
        )

    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403, detail="Admin access required."
        )

    return True
