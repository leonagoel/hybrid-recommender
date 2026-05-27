"""
Unit and integration tests for NLP issue triage classifier, rule overrides,
assignee suggestor, and GitHub webhook event API integration.
"""

import pytest
import hmac
import hashlib
import os
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from src.model.issue_triage import IssueClassifier, get_suggested_assignees, format_triage_comment, triage_issue
from backend import main


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_issue_classifier_nlp_and_rules():
    classifier = IssueClassifier()
    
    # 1. Test ML Domain prediction
    res_ml = classifier.predict("Build a collaborative filtering recommender system", "")
    assert res_ml["domain"]["label"] == "ml"
    
    # 2. Test Frontend Domain prediction
    res_fe = classifier.predict("CSS styling issues on footer layout alignment", "")
    assert res_fe["domain"]["label"] == "frontend"
    
    # 3. Test Backend Domain prediction
    res_be = classifier.predict("Create database connection pools for FastAPI backend server", "")
    assert res_be["domain"]["label"] == "backend"
    
    # 4. Test Security rule override
    res_sec = classifier.predict("Fix critical SQL injection vulnerability in search endpoint", "")
    assert res_sec["type"]["label"] == "security"
    assert res_sec["level"]["label"] == "critical"
    assert res_sec["priority"]["label"] == "critical"
    
    # 5. Test Beginner / doc indicators
    res_beg = classifier.predict("Update README documentation typo in setup guide", "")
    assert res_beg["level"]["label"] == "beginner"
    assert res_beg["priority"]["label"] == "low"


def test_get_suggested_assignees():
    assert "ml-expert-dev" in get_suggested_assignees("ml")
    assert "ui-designer-dev" in get_suggested_assignees("frontend")
    assert "backend-core-dev" in get_suggested_assignees("backend")
    assert get_suggested_assignees("other") == []


def test_format_triage_comment():
    predictions = {
        "type": {"label": "bug", "confidence": 0.9, "reason": "Test reasoning"},
        "domain": {"label": "frontend", "confidence": 0.8, "reason": "Test reasoning"},
        "level": {"label": "beginner", "confidence": 0.7, "reason": "Test reasoning"},
        "priority": {"label": "low", "confidence": 0.6, "reason": "Test reasoning"},
    }
    comment = format_triage_comment(predictions, ["dev1"])
    assert "type:bug" in comment
    assert "frontend" in comment
    assert "level:beginner" in comment
    assert "priority:low" in comment
    assert "@dev1" in comment


@pytest.mark.anyio
async def test_triage_issue_skips_api_if_no_token(monkeypatch):
    # If no token is provided, triage_issue should skip GitHub API calls
    res = await triage_issue(
        issue_number=100,
        title="Test issue",
        body="Frontend button misalignment",
        repo_full_name="org/repo",
        token=""
    )
    assert res["issue_number"] == 100
    assert res["predictions"]["domain"]["label"] == "frontend"
    assert res["github_api"]["status"] == "skipped"


@pytest.mark.anyio
async def test_triage_issue_calls_api_if_token(monkeypatch):
    # Mock apply_github_actions
    mock_apply = AsyncMock(return_value={"labels": 200, "comment": 201})
    monkeypatch.setattr("src.model.issue_triage.apply_github_actions", mock_apply)
    
    res = await triage_issue(
        issue_number=200,
        title="SQL Injection vulnerability",
        body="Severe security leak",
        repo_full_name="org/repo",
        token="fake-token"
    )
    
    assert res["issue_number"] == 200
    assert res["predictions"]["type"]["label"] == "security"
    assert res["github_api"] == {"labels": 200, "comment": 201}
    mock_apply.assert_called_once()


def test_webhook_signature_verification(monkeypatch):
    # Set GITHUB_WEBHOOK_SECRET in environment
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "super-secret-key")
    client = TestClient(main.app)
    
    # 1. No signature header should return 401
    res_no_sig = client.post("/api/webhook/github", json={"action": "opened"})
    assert res_no_sig.status_code == 401
    
    # 2. Invalid signature header should return 403
    headers = {"X-GitHub-Event": "issues", "X-Hub-Signature-256": "sha256=invalid-signature"}
    res_bad_sig = client.post("/api/webhook/github", json={"action": "opened"}, headers=headers)
    assert res_bad_sig.status_code == 403
    
    # 3. Valid signature should work (or return 200/skipped if event not supported)
    payload = {"action": "opened", "issue": {"number": 1, "title": "t", "body": "b"}, "repository": {"full_name": "o/r"}}
    import json
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    # Compute signature
    sig = hmac.new(b"super-secret-key", payload_bytes, hashlib.sha256).hexdigest()
    headers_valid = {
        "X-GitHub-Event": "issues",
        "X-Hub-Signature-256": f"sha256={sig}"
    }
    
    # Mock triage_issue
    mock_triage = AsyncMock(return_value={"status": "mocked"})
    monkeypatch.setattr("backend.main.triage_issue", mock_triage)
    
    res_valid = client.post("/api/webhook/github", content=payload_bytes, headers=headers_valid)
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "success"


def test_webhook_event_routing(monkeypatch):
    # Disable webhook secret check by clearing env (or default is empty)
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    client = TestClient(main.app)
    
    mock_triage = AsyncMock(return_value={"status": "mocked"})
    monkeypatch.setattr("backend.main.triage_issue", mock_triage)
    
    # 1. Issues opened event
    payload_opened = {
        "action": "opened",
        "issue": {"number": 12, "title": "Feature addition", "body": "Need SVD model support"},
        "repository": {"full_name": "owner/repo"}
    }
    res_opened = client.post(
        "/api/webhook/github",
        json=payload_opened,
        headers={"X-GitHub-Event": "issues"}
    )
    assert res_opened.status_code == 200
    assert res_opened.json()["status"] == "success"
    mock_triage.assert_called_with(
        issue_number=12,
        title="Feature addition",
        body="Need SVD model support",
        repo_full_name="owner/repo",
        token=""
    )
    
    # Reset mock call history
    mock_triage.reset_mock()
    
    # 2. Issue comment retriage event
    payload_comment = {
        "action": "created",
        "comment": {"body": " !retriage please "},
        "issue": {"number": 12, "title": "Feature addition", "body": "Need SVD model support"},
        "repository": {"full_name": "owner/repo"}
    }
    res_comment = client.post(
        "/api/webhook/github",
        json=payload_comment,
        headers={"X-GitHub-Event": "issue_comment"}
    )
    assert res_comment.status_code == 200
    assert res_comment.json()["status"] == "success"
    mock_triage.assert_called_once()
    
    # Reset mock call history
    mock_triage.reset_mock()
    
    # 3. Unrelated comment event (should be skipped)
    payload_other_comment = {
        "action": "created",
        "comment": {"body": "I want to work on this issue"},
        "issue": {"number": 12, "title": "Feature addition", "body": "Need SVD model support"},
        "repository": {"full_name": "owner/repo"}
    }
    res_other = client.post(
        "/api/webhook/github",
        json=payload_other_comment,
        headers={"X-GitHub-Event": "issue_comment"}
    )
    assert res_other.status_code == 200
    assert res_other.json()["status"] == "skipped"
    mock_triage.assert_not_called()
