import os

from fastapi import Header, HTTPException


ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")


def _require_admin_access(x_admin_key: str = Header(default=None)):
    """
    Require a valid admin API key for protected endpoints.
    """

    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY is not configured on the server."
        )

    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Admin access required."
        )

    return True