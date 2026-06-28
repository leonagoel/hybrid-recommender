"""
tests/test_jwt_auth.py — Tests for Supabase JWT verification (issue #297).

Background: the backend previously had NO JWT verification at all.
Routes like /api/recommend/user/{user_id} and /api/purchases/{user_id}
trusted whatever user_id appeared in the URL, and /api/v1/user/preferences/reset
trusted a plain (spoofable) x-user-id header. Supabase's own access tokens
already carry a real `exp` claim and are refreshed client-side — the gap
was that the backend never checked the signature or expiry at all.

Covers:
  - A validly-signed, unexpired token is accepted and its `sub` returned.
  - An expired token is rejected with 401 (the literal bug in #297).
  - A token signed with the wrong secret (forged/tampered) is rejected with 401.
  - A token missing the `Authorization` header is rejected with 401.
  - A token with the wrong `aud` claim is rejected with 401.
  - A token missing the `sub` claim is rejected with 401.
  - Misconfiguration (no SUPABASE_JWT_SECRET set) fails closed with 500,
    never by silently accepting unverified tokens.
  - An authenticated route only serves the token's own user_id (ownership
    check), rejecting requests for another user's data with 403.
"""

import os
import time

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("TESTING", "true")

TEST_SECRET = "test-supabase-jwt-secret-for-unit-tests-only"

# Set this before importing backend.auth so the module-level
# SUPABASE_JWT_SECRET constant picks it up.
os.environ["SUPABASE_JWT_SECRET"] = TEST_SECRET
os.environ["SUPABASE_JWT_AUDIENCE"] = "authenticated"

from backend import auth as auth_module  # noqa: E402  (after env setup)


def _make_token(
    sub: str = "user_123",
    secret: str = TEST_SECRET,
    aud: str = "authenticated",
    expires_in_seconds: int = 3600,
    algorithm: str = "HS256",
) -> str:
    """Build a Supabase-shaped JWT for testing."""
    now = int(time.time())
    payload = {
        "sub": sub,
        "aud": aud,
        "iat": now,
        "exp": now + expires_in_seconds,
        "role": "authenticated",
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


@pytest.fixture(autouse=True)
def _reset_secret():
    """Ensure each test starts from the same known-good secret/audience."""
    auth_module.SUPABASE_JWT_SECRET = TEST_SECRET
    auth_module.SUPABASE_JWT_AUDIENCE = "authenticated"
    yield
    auth_module.SUPABASE_JWT_SECRET = TEST_SECRET
    auth_module.SUPABASE_JWT_AUDIENCE = "authenticated"


# ── Unit tests: verify_supabase_jwt ────────────────────────────────────────────

def test_valid_unexpired_token_is_accepted():
    token = _make_token(sub="user_abc", expires_in_seconds=3600)
    claims = auth_module.verify_supabase_jwt(token)
    assert claims["sub"] == "user_abc"


def test_expired_token_is_rejected_with_401():
    """The literal bug reported in #297: an expired token must not be accepted."""
    token = _make_token(sub="user_abc", expires_in_seconds=-60)  # expired 60s ago
    with pytest.raises(Exception) as exc_info:
        auth_module.verify_supabase_jwt(token)
    assert getattr(exc_info.value, "status_code", None) == 401
    assert "expired" in str(exc_info.value.detail).lower()


def test_token_signed_with_wrong_secret_is_rejected():
    """A forged/tampered token (wrong signing key) must be rejected, not just expiry."""
    token = _make_token(sub="user_abc", secret="some-other-secret-entirely")
    with pytest.raises(Exception) as exc_info:
        auth_module.verify_supabase_jwt(token)
    assert getattr(exc_info.value, "status_code", None) == 401


def test_token_with_wrong_audience_is_rejected():
    token = _make_token(sub="user_abc", aud="some-other-audience")
    with pytest.raises(Exception) as exc_info:
        auth_module.verify_supabase_jwt(token)
    assert getattr(exc_info.value, "status_code", None) == 401


def test_missing_token_is_rejected():
    with pytest.raises(Exception) as exc_info:
        auth_module.verify_supabase_jwt("")
    assert getattr(exc_info.value, "status_code", None) == 401


def test_token_missing_sub_claim_is_rejected_by_get_current_user_id():
    now = int(time.time())
    payload = {"aud": "authenticated", "iat": now, "exp": now + 3600}
    token = jwt.encode(payload, TEST_SECRET, algorithm="HS256")

    app = FastAPI()

    @app.get("/whoami")
    def whoami(user_id: str = Depends(auth_module.get_current_user_id)):
        return {"user_id": user_id}

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_misconfigured_secret_fails_closed_not_open():
    """If SUPABASE_JWT_SECRET is unset, we must refuse to verify (500),
    never silently treat the token as valid."""
    auth_module.SUPABASE_JWT_SECRET = ""
    token = _make_token(sub="user_abc")
    with pytest.raises(Exception) as exc_info:
        auth_module.verify_supabase_jwt(token)
    assert getattr(exc_info.value, "status_code", None) == 500


# ── Integration tests: get_current_user_id as a FastAPI dependency ────────────

@pytest.fixture()
def whoami_app():
    """A minimal app exercising get_current_user_id exactly as backend/main.py does."""
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user_id: str = Depends(auth_module.get_current_user_id)):
        return {"user_id": user_id}

    return app


@pytest.fixture()
def whoami_client(whoami_app):
    with TestClient(whoami_app, raise_server_exceptions=False) as c:
        yield c


def test_dependency_returns_subject_for_valid_token(whoami_client):
    token = _make_token(sub="user_42")
    response = whoami_client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user_id"] == "user_42"


def test_dependency_rejects_expired_token(whoami_client):
    token = _make_token(sub="user_42", expires_in_seconds=-1)
    response = whoami_client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_dependency_rejects_missing_authorization_header(whoami_client):
    response = whoami_client.get("/whoami")
    assert response.status_code == 401


def test_dependency_rejects_non_bearer_scheme(whoami_client):
    token = _make_token(sub="user_42")
    response = whoami_client.get("/whoami", headers={"Authorization": f"Token {token}"})
    assert response.status_code == 401


# ── Ownership check pattern used in backend/main.py routes ────────────────────

@pytest.fixture()
def owned_resource_app():
    """
    Mirrors the pattern used in get_user_recommendations / get_user_purchases:
    the path user_id must match the verified token's subject, or 403.
    """
    app = FastAPI()

    @app.get("/resource/{user_id}")
    def get_resource(user_id: str, authenticated_user_id: str = Depends(auth_module.get_current_user_id)):
        if authenticated_user_id != user_id:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Cannot access another user's data.")
        return {"user_id": user_id, "data": "secret"}

    return app


@pytest.fixture()
def owned_resource_client(owned_resource_app):
    with TestClient(owned_resource_app, raise_server_exceptions=False) as c:
        yield c


def test_user_can_access_own_resource(owned_resource_client):
    token = _make_token(sub="user_42")
    response = owned_resource_client.get(
        "/resource/user_42", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200


def test_user_cannot_access_another_users_resource(owned_resource_client):
    token = _make_token(sub="user_42")
    response = owned_resource_client.get(
        "/resource/someone_else", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403


def test_expired_token_cannot_access_resource_even_if_ids_match(owned_resource_client):
    """Belt-and-suspenders: even a correct user_id match must not bypass expiry."""
    token = _make_token(sub="user_42", expires_in_seconds=-1)
    response = owned_resource_client.get(
        "/resource/user_42", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
    