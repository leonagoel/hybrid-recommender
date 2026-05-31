import os
import hashlib
import hmac
from typing import Optional

from fastapi import Header, HTTPException


TOKEN_COMPARE_CONTEXT = b"hybrid-recommender-admin-token-v1"


def _token_digest(token: Optional[str]) -> bytes:
    """Return a fixed-length digest so equality checks never compare raw secrets."""
    normalized = "" if token is None else str(token)
    return hmac.new(
        TOKEN_COMPARE_CONTEXT,
        normalized.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def constant_time_token_equal(
    provided_token: Optional[str],
    expected_token: Optional[str],
) -> bool:
    if not provided_token or not expected_token:
        return False

    provided_digest = _token_digest(provided_token)
    expected_digest = _token_digest(expected_token)
    return hmac.compare_digest(provided_digest, expected_digest)


def _require_admin_access(x_admin_key: str = Header(default=None)):
    """
    Require a valid admin API key for protected endpoints.
    """

    admin_api_key = os.environ.get("ADMIN_API_KEY")
    if not admin_api_key:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY is not configured on the server."
        )

    if not constant_time_token_equal(x_admin_key, admin_api_key):
        raise HTTPException(
            status_code=403,
            detail="Admin access required."
        )

    return True
