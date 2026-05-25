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


def test_admin_access_allows_when_token_not_configured(monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)

    _require_admin_access(FakeRequest())


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
