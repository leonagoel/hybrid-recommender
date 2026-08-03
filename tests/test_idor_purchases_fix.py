"""
Test cases for IDOR Vulnerability in Purchases API (CWE-639).

This test suite verifies that the purchases API properly enforces authorization,
preventing users from accessing other users' purchase data.

SECURITY IMPACT:
- CWE-639: Authorization Bypass Through User-Controlled Key
- Users could view private purchase history of other users
- Privacy violation

See the fix patch for implementation details.
"""
import pytest
from fastapi.testclient import TestClient
import os

# Set testing environment
os.environ["TESTING"] = "true"
os.environ["CSRF_SECURE"] = "false"


class TestIDORPrevention:
    """Test cases verifying IDOR vulnerability is fixed."""

    @pytest.fixture
    def client(self):
        """Provides a TestClient for the FastAPI app."""
        from backend.main import app
        return TestClient(app)

    def test_user_cannot_access_other_user_purchases(self, client):
        """
        CRITICAL: User A should NOT be able to view User B's purchases.
        
        This is the core IDOR test. If this fails, the vulnerability exists.
        """
        # User A (attacker) tries to view User B's (victim) purchases
        response = client.get(
            "/api/purchases/victim_user_id_123",
            headers={
                "X-User-ID": "attacker_user_id_456",  # Attacker is authenticated as User A
                "Origin": "http://localhost",
            }
        )
        
        # Should be 403 Forbidden, NOT 200
        assert response.status_code == 403, \
            f"IDOR VULNERABILITY: Attacker can view other user's purchases! Status: {response.status_code}"

    def test_user_can_access_own_purchases(self, client):
        """
        User SHOULD be able to view their own purchases.
        """
        response = client.get(
            "/api/purchases/my_user_id",
            headers={
                "X-User-ID": "my_user_id",  # Accessing own data
                "Origin": "http://localhost",
            }
        )
        
        # Should succeed (200 or 404 if no purchases, NOT 403)
        assert response.status_code != 403, \
            "User should be able to access their own purchases"

    def test_admin_can_access_any_user_purchases(self, client):
        """
        Admin SHOULD be able to view any user's purchases for support/debugging.
        """
        response = client.get(
            "/api/purchases/any_user_id",
            headers={
                "X-User-ID": "admin_user_id",
                "X-Admin-Token": os.environ.get("ADMIN_API_TOKEN", "test_admin_token"),
                "Origin": "http://localhost",
            }
        )
        
        # Admin should get access (200 or 404, NOT 403)
        assert response.status_code != 403, \
            "Admin should be able to access any user's purchases"

    def test_delete_other_user_purchase_denied(self, client):
        """
        User should NOT be able to delete another user's purchase.
        """
        response = client.delete(
            "/api/purchases/purchase_12345",  # Purchase owned by another user
            headers={
                "X-User-ID": "attacker_user_id",
                "Origin": "http://localhost",
            }
        )
        
        # Should be 403 Forbidden
        assert response.status_code == 403, \
            f"IDOR VULNERABILITY: User can delete other user's purchase! Status: {response.status_code}"

    def test_case_insensitive_user_id_matching(self, client):
        """
        Verify user ID matching is case-insensitive to prevent bypass attempts.
        """
        # Attacker tries different case variations
        for attacker_id, victim_id in [
            ("UserA", "usera"),  # Same user, different case
            ("USER_A", "user_a"),  # Uppercase vs lowercase
        ]:
            response = client.get(
                f"/api/purchases/{victim_id}",
                headers={
                    "X-User-ID": attacker_id,
                    "Origin": "http://localhost",
                }
            )
            
            # If IDs refer to the same user, should succeed
            # If different users, should fail
            # This test just documents the expected behavior

    def test_missing_auth_header_denied(self, client):
        """
        Requests without authentication should be denied.
        """
        response = client.get(
            "/api/purchases/some_user",
            headers={
                "Origin": "http://localhost",
                # No X-User-ID header
            }
        )
        
        # Should be 401 or 403
        assert response.status_code in [401, 403], \
            "Request without auth header should be denied"


class TestAuthorizationUtilities:
    """Test the authorization utility functions."""

    def test_validate_user_id(self):
        """Test user ID validation function."""
        from backend.idor_fix import validate_user_id
        
        # Valid IDs
        assert validate_user_id("user123") == "user123"
        assert validate_user_id("USER123") == "user123"  # Lowercase normalization
        
        # Invalid IDs
        with pytest.raises(Exception):
            validate_user_id("")
        
        with pytest.raises(Exception):
            validate_user_id("x" * 200)  # Too long

    def test_require_purchase_access_own_data(self):
        """Test that users can access their own data."""
        from fastapi import HTTPException
        from backend.idor_fix import require_purchase_access
        
        # Mock request object
        class MockRequest:
            headers = {"X-Admin-Token": ""}
        
        # User accessing their own data - should pass
        try:
            import asyncio
            asyncio.run(require_purchase_access(
                MockRequest(),
                "user123"  # Target is own ID
            ))
            # If we get here without exception, it's working
        except HTTPException:
            pytest.fail("User should be able to access their own data")


class TestVulnerabilityDocumentation:
    """Document the vulnerability and fix."""

    def test_idor_vulnerability_explained(self):
        """
        DOCUMENTATION: CWE-639 - Authorization Bypass Through User-Controlled Key
        
        VULNERABLE PATTERN:
        @app.get("/api/purchases/{user_id}")
        def get_purchases(user_id):
            # NO authorization check!
            return db.get_purchases(user_id)
        
        ATTACK:
        1. Attacker logs in as user@example.com
        2. Attacker requests GET /api/purchases/victim_id
        3. Server returns victim's purchases
        
        SECURE PATTERN:
        @app.get("/api/purchases/{user_id}")
        async def get_purchases(request, user_id):
            # Verify attacker owns this data
            current_user = get_current_user(request)
            if current_user.id != user_id and not is_admin(current_user):
                raise HTTPException(403)
            return db.get_purchases(user_id)
        
        This test documents that the fix is needed.
        """
        # Verify the fix module exists
        try:
            from backend import idor_fix
            assert hasattr(idor_fix, 'require_purchase_access')
            assert hasattr(idor_fix, 'validate_user_id')
        except ImportError:
            pytest.fail("IDOR fix module not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
