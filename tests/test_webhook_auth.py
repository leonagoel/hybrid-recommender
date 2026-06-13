import pytest
from fastapi.testclient import TestClient
from backend.main import app, _verify_github_signature
from fastapi import HTTPException
import os

def test_verify_github_signature_missing_secret(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "")
    with pytest.raises(HTTPException) as exc_info:
        _verify_github_signature(b"{}", "sha256=dummy")
    assert exc_info.value.status_code == 500
    assert "GITHUB_WEBHOOK_SECRET is not configured" in exc_info.value.detail

def test_verify_github_signature_missing_header(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "mysecret")
    with pytest.raises(HTTPException) as exc_info:
        _verify_github_signature(b"{}", None)
    assert exc_info.value.status_code == 401
    assert "Signature header" in exc_info.value.detail

def test_verify_github_signature_invalid_format(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "mysecret")
    with pytest.raises(HTTPException) as exc_info:
        _verify_github_signature(b"{}", "dummy")
    assert exc_info.value.status_code == 400
    assert "Invalid signature format" in exc_info.value.detail

def test_verify_github_signature_invalid_signature(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "mysecret")
    with pytest.raises(HTTPException) as exc_info:
        _verify_github_signature(b"{}", "sha256=dummy")
    assert exc_info.value.status_code == 403
    assert "Invalid webhook signature" in exc_info.value.detail
