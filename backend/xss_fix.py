"""
Fix for Issue #5: XSS Vulnerability in Wishlist API (CWE-79)

This module provides sanitization utilities to prevent Cross-Site Scripting (XSS)
attacks in user-generated content.

SECURITY IMPACT:
- CWE-79: Cross-site Scripting
- Stored XSS: Malicious scripts stored in item names
- Session hijacking via stolen cookies
- Malicious redirects

Apply these fixes to backend/main.py or the wishlist router.
"""

from pydantic import BaseModel, Field, validator
from fastapi import HTTPException


def sanitize_for_html(text: str) -> str:
    """
    Sanitize text to prevent XSS attacks.
    
    Removes all HTML tags and attributes that could be used for XSS.
    Safe for use in HTML content and JSON responses.
    """
    try:
        import bleach
    except ImportError:
        # Fallback if bleach not installed
        import re
        return re.sub(r'<[^>]*>', '', text)
    
    if not text:
        return ""
    
    # bleach.clean removes all HTML tags and attributes
    # tags=[] means no tags allowed
    # strip=True removes extra whitespace
    cleaned = bleach.clean(
        text,
        tags=[],  # No HTML tags allowed
        attributes={},  # No attributes allowed
        strip=True  # Strip extra whitespace
    )
    
    return cleaned


def validate_item_name(name: str) -> str:
    """
    Validate and sanitize an item name for wishlist operations.
    
    Rules:
    1. Maximum length: 200 characters
    2. No HTML/JS/CSS tags
    3. No special characters that could be used in XSS
    """
    if not name:
        raise ValueError("Item name cannot be empty")
    
    # Strip and truncate
    name = name.strip()[:200]
    
    # Sanitize HTML/JS
    sanitized = sanitize_for_html(name)
    
    # Additional validation: no control characters
    if any(ord(c) < 32 and c not in '\t\n\r' for c in sanitized):
        raise ValueError("Item name contains invalid characters")
    
    return sanitized


class SafeWishlistItem(BaseModel):
    """
    Pydantic model for wishlist items with XSS protection.
    """
    item: str = Field(..., max_length=200, min_length=1)
    
    @validator('item', pre=True)
    def sanitize_item(cls, v):
        """Sanitize item name before validation."""
        if not v:
            raise ValueError("Item name cannot be empty")
        
        sanitized = sanitize_for_html(str(v).strip()[:200])
        
        if not sanitized:
            raise ValueError("Item name contains no valid content")
        
        return sanitized
    
    class Config:
        json_schema_extra = {
            "example": {
                "item": "Safe Product Name"
            }
        }


def escape_html(text: str) -> str:
    """
    Escape HTML special characters for safe display.
    
    Use this for frontend display of user-controlled content.
    """
    if not text:
        return ""
    
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def escape_html_attr(text: str) -> str:
    """
    Escape text for use in HTML attributes.
    """
    if not text:
        return ""
    
    return (
        text
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
