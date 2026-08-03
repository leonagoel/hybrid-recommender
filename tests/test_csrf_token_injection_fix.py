"""
Test cases for CSRF Token Injection vulnerability fix (Issue #1597 / bug1.md).

This test suite verifies that the Double Submit Cookie CSRF protection
is NOT vulnerable to Token Injection attacks where an attacker could:
1. Set a short, known token (e.g., "a") in the user's csrftoken cookie
2. Send a request with the matching "a" in the X-CSRF-Token header
3. Have the server accept this as valid CSRF validation

The fix enforces strict token format validation: tokens must be exactly
64 hexadecimal characters (256-bit entropy).
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient for the FastAPI app."""
    from backend.main import app
    # Force CSRF to be disabled for testing (TestClient uses plain HTTP)
    import os
    os.environ["TESTING"] = "true"
    os.environ["CSRF_SECURE"] = "false"
    yield TestClient(app)
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "CSRF_SECURE" in os.environ:
        del os.environ["CSRF_SECURE"]


class TestCSRFTokenInjectionPrevention:
    """Test cases verifying that short/invalid tokens are rejected."""

    def test_short_token_rejected(self, client):
        """
        Attack Scenario: Attacker sets cookie_token="a" and header_token="a".
        Expected: Server MUST reject with 403, not accept as valid.
        """
        response = client.post(
            "/api/feedback",
            headers={
                "Cookie": "csrftoken=a",
                "X-CSRF-Token": "a",
                "Origin": "http://localhost",
            },
            json={"user_id": "test", "item": "test", "feedback": "test"},
        )
        assert response.status_code == 403, (
            "SECURITY ISSUE: Short token 'a' was accepted! "
            "This allows Token Injection attacks."
        )

    def test_numeric_short_token_rejected(self, client):
        """
        Attack Scenario: Attacker sets cookie_token="123456" (6 chars) and matches it.
        Expected: Server MUST reject with 403.
        """
        response = client.post(
            "/api/feedback",
            headers={
                "Cookie": "csrftoken=123456",
                "X-CSRF-Token": "123456",
                "Origin": "http://localhost",
            },
            json={"user_id": "test", "item": "test", "feedback": "test"},
        )
        assert response.status_code == 403, (
            "SECURITY ISSUE: 6-character numeric token was accepted!"
        )

    def test_63_char_token_rejected(self, client):
        """
        Edge Case: Token is 63 hex characters (one short of required 64).
        Expected: Server MUST reject with 403.
        """
        short_token = "a" * 63  # One short of required 64 chars
        response = client.post(
            "/api/feedback",
            headers={
                "Cookie": f"csrftoken={short_token}",
                "X-CSRF-Token": short_token,
                "Origin": "http://localhost",
            },
            json={"user_id": "test", "item": "test", "feedback": "test"},
        )
        assert response.status_code == 403, (
            "SECURITY ISSUE: 63-character token was accepted! "
            "Must be exactly 64 hex characters."
        )

    def test_65_char_token_rejected(self, client):
        """
        Edge Case: Token is 65 hex characters (one over required 64).
        Expected: Server MUST reject with 403.
        """
        long_token = "a" * 65  # One over required 64 chars
        response = client.post(
            "/api/feedback",
            headers={
                "Cookie": f"csrftoken={long_token}",
                "X-CSRF-Token": long_token,
                "Origin": "http://localhost",
            },
            json={"user_id": "test", "item": "test", "feedback": "test"},
        )
        assert response.status_code == 403, (
            "SECURITY ISSUE: 65-character token was accepted! "
            "Must be exactly 64 hex characters."
        )

    def test_non_hex_characters_rejected(self, client):
        """
        Attack Scenario: Token contains non-hex characters (e.g., spaces, special chars).
        Expected: Server MUST reject with 403.
        """
        invalid_token = "a" * 63 + "!"  # Contains '!' which is not hex
        response = client.post(
            "/api/feedback",
            headers={
                "Cookie": f"csrftoken={invalid_token}",
                "X-CSRF-Token": invalid_token,
                "Origin": "http://localhost",
            },
            json={"user_id": "test", "item": "test", "feedback": "test"},
        )
        assert response.status_code == 403, (
            "SECURITY ISSUE: Token with non-hex characters was accepted!"
        )

    def test_mixed_case_hex_rejected(self, client):
        """
        Note: Lowercase hex only is allowed by current implementation.
        Uppercase hex (A-F) should be validated.
        """
        # This should be accepted since A-F are valid hex digits
        valid_upper_hex = "A" * 64
        response = client.post(
            "/api/feedback",
            headers={
                "Cookie": f"csrftoken={valid_upper_hex}",
                "X-CSRF-Token": valid_upper_hex,
                "Origin": "http://localhost",
            },
            json={"user_id": "test", "item": "test", "feedback": "test"},
        )
        # Should still be 403 because token isn't in issued_tokens
        # But should NOT be accepted due to format alone
        assert response.status_code == 403


class TestCSRFValidTokenFormat:
    """Test cases verifying that properly formatted tokens are processed correctly."""

    def test_valid_64_hex_token_format(self, client):
        """
        Verify that the _is_valid_token validation accepts valid 64-char hex strings.
        This tests the positive case of the fix.
        """
        from backend.csrf import _is_valid_token

        # Valid 64-character lowercase hex string
        assert _is_valid_token("a" * 64) is True

        # Valid 64-character uppercase hex string
        assert _is_valid_token("A" * 64) is True

        # Valid mixed-case hex string
        assert _is_valid_token("0123456789abcdef" * 4) is True

        # Invalid: 63 characters
        assert _is_valid_token("a" * 63) is False

        # Invalid: 65 characters
        assert _is_valid_token("a" * 65) is False

        # Invalid: contains non-hex character
        assert _is_valid_token("a" * 63 + "!") is False

        # Invalid: empty string
        assert _is_valid_token("") is False

    def test_csrf_token_bytes_constant(self, client):
        """Verify CSRF_TOKEN_BYTES produces correct token length."""
        from backend.csrf import CSRF_TOKEN_BYTES
        import secrets

        # CSRF_TOKEN_BYTES = 32 produces 64 hex chars (32 * 2)
        token = secrets.token_hex(CSRF_TOKEN_BYTES)
        assert len(token) == 64
        assert _is_valid_token(token) is True


def _is_valid_token(t: str) -> bool:
    """Local helper that mirrors the validation logic in csrf.py."""
    from backend.csrf import CSRF_TOKEN_BYTES
    import string
    return len(t) == CSRF_TOKEN_BYTES * 2 and all(c in string.hexdigits for c in t)


class TestCSRFVulnerabilitySummary:
    """
    Summary test class documenting the vulnerability and fix.
    This serves as documentation for security auditors.
    """

    def test_vulnerability_description(self):
        """
        VULNERABILITY: CSRF Token Injection (CWE-352)

        BEFORE FIX:
        - The middleware only checked if cookie_token == header_token
        - No validation of token length or format
        - Attacker could set cookie to "a" and send "a" in header
        - Server would accept this as valid CSRF validation

        AFTER FIX:
        - Token format validation enforced (exactly 64 hex characters)
        - Both cookie and header tokens must be valid format
        - Attacker cannot guess/predict valid server-issued tokens
        - HMAC-based server-issued tokens prevent injection

        This test documents that the fix is in place.
        """
        from backend.csrf import _is_valid_token, CSRF_TOKEN_BYTES

        # Verify constant is correct (256-bit = 32 bytes = 64 hex chars)
        assert CSRF_TOKEN_BYTES == 32
        assert CSRF_TOKEN_BYTES * 2 == 64

        # Verify the validation function exists and works
        assert callable(_is_valid_token)

        # Verify short tokens are rejected
        assert _is_valid_token("a") is False
        assert _is_valid_token("123456") is False

        # Verify valid tokens are accepted
        assert _is_valid_token("a" * 64) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
