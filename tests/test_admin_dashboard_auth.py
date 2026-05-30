import pytest
from fastapi import HTTPException

from backend.main import _extract_bearer_token, _require_admin_access


class FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_extract_bearer_token_accepts_bearer_only():
    assert _extract_bearer_token("Bearer secret-token") == "secret-token"
    assert _extract_bearer_token("Basic secret-token") == ""
    assert _extract_bearer_token(None) == ""


def test_admin_access_fails_when_token_not_configured(monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)

    with pytest.raises(HTTPException) as exc:
        _require_admin_access(FakeRequest())

    assert exc.value.status_code == 500
    assert exc.value.detail == "Admin token not configured."


@pytest.mark.parametrize(
    "headers",
    [
        {"x-admin-token": "expected-token"},
        {"authorization": "Bearer expected-token"},
    ],
)
def test_admin_access_accepts_configured_token(monkeypatch, headers):
    monkeypatch.setenv("ADMIN_API_TOKEN", "expected-token")

    _require_admin_access(FakeRequest(headers))


def test_admin_access_rejects_missing_or_invalid_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "expected-token")

    with pytest.raises(HTTPException) as exc:
        _require_admin_access(FakeRequest({"x-admin-token": "wrong-token"}))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Admin token required."


# ── Upload endpoint admin auth integration ──────────────────────────────

def test_upload_endpoint_rejects_without_admin_token(monkeypatch):
    """POST /api/upload must reject with 401 when ADMIN_API_TOKEN is set and
    no admin token header is provided, even with valid CSRF."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "test-admin-token")

    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # Get a fresh CSRF token
    csrf_resp = client.get("/api/csrf-token")
    token = csrf_resp.json()["csrfToken"]

    payload = {"file": ("test.csv", b"title,rating\nAlpha,5", "text/csv")}
    response = client.post(
        "/api/upload",
        files=payload,
        headers={"x-csrf-token": token},
    )
    # Without admin token → 401 from _require_admin_access
    assert response.status_code == 401
    assert "Admin token required" in response.json()["detail"]


def test_upload_endpoint_allows_with_admin_token(monkeypatch):
    """POST /api/upload with valid CSRF + admin token must pass admin auth.
    The request may fail later (e.g. Supabase not configured), but must NOT
    receive a 401 or a 500 with 'Admin token not configured'."""
    monkeypatch.setenv("ADMIN_API_TOKEN", "test-admin-token")

    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    csrf_resp = client.get("/api/csrf-token")
    token = csrf_resp.json()["csrfToken"]

    payload = {"file": ("test.csv", b"title,rating\nAlpha,5", "text/csv")}
    response = client.post(
        "/api/upload",
        files=payload,
        headers={
            "x-csrf-token": token,
            "x-admin-token": "test-admin-token",
        },
    )
    # Must NOT be admin auth failure responses
    assert response.status_code not in (401,)
    if response.status_code == 500:
        assert "Admin token" not in response.json().get("detail", "")
