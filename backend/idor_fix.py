"""
Fix for Issue #4: IDOR Vulnerability in Purchases API (CWE-639)

This module provides authorization utilities to prevent Insecure Direct Object
Reference (IDOR) attacks in the purchases API.

SECURITY IMPACT:
- Users can view other users' private purchase data
- Privacy violation
- Potential data leakage

Apply these fixes to backend/main.py or the purchases router.
"""

from fastapi import Header, HTTPException, Request
import re
import os


def validate_user_id(user_id: str) -> str:
    """
    Validate user ID format and normalize it.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Normalize to lowercase for consistent comparison
    normalized = user_id.lower().strip()
    
    if len(normalized) > 100:
        raise HTTPException(status_code=400, detail="User ID too long")
    
    return normalized


async def get_current_user_id(x_user_id: str = Header(..., alias="X-User-ID")) -> str:
    """
    Extract and validate the current user ID from request headers.
    
    In production, this should validate a JWT or session token.
    For now, we use the X-User-ID header with format validation.
    """
    return validate_user_id(x_user_id)


async def require_purchase_access(
    request: Request,
    target_user_id: str,
) -> None:
    """
    Verify the requesting user has access to the target user's purchase data.
    
    Rules:
    1. Users can only access their own purchase data
    2. Admins can access any user's purchase data
    
    Raises HTTPException 403 if access is denied.
    """
    current_user = await get_current_user_id()
    normalized_target = validate_user_id(target_user_id)
    
    # Check if user is accessing their own data
    if current_user == normalized_target:
        return  # Access granted
    
    # Check if user has admin privileges
    admin_token = request.headers.get("X-Admin-Token", "")
    expected_admin_token = os.environ.get("ADMIN_API_TOKEN", "")
    
    if admin_token and admin_token == expected_admin_token:
        return  # Admin access granted
    
    # Deny access
    raise HTTPException(
        status_code=403,
        detail="Not authorized to access this user's purchase data"
    )


def get_current_user_or_none(request: Request) -> str | None:
    """
    Get the current user ID if authenticated, None otherwise.
    """
    try:
        return get_current_user_id(request)
    except HTTPException:
        return None
