import os
import secrets
from fastapi import Request, HTTPException
from backend.core.config import ADMIN_API_TOKEN_ENV

def _extract_bearer_token(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()

def _require_admin_access(request: Request) -> None:
    expected_token = os.environ.get(ADMIN_API_TOKEN_ENV, "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Admin token not configured.",
        )

    provided_token = (
        request.headers.get("x-admin-token", "").strip()
        or _extract_bearer_token(request.headers.get("authorization"))
    )
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=401, detail="Admin token required.")

def _admin_access_dep(request: Request) -> None:
    _require_admin_access(request)
