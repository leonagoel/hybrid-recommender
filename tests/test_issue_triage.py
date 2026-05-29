"""
Unit and integration tests for NLP issue triage classifier, rule overrides,
assignee suggestor, and GitHub webhook event API integration.
"""

import pytest
import hmac
import hashlib
import os
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.model.issue_triage import IssueClassifier, get_suggested_assignees, format_triage_comment, triage_issue
from backend import main


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ===========================================================================
# EXTENDED EDGE-CASE TESTS FOR ISSUE #625
# ===========================================================================

def test_issue_classifier_clean_text_edge_cases():
    classifier = IssueClassifier()
    
    # Test extreme whitespaces, line breaks, tabs, and messy casing
    messy_text = "   \n\n  FEatURE:  centralized   dotenv    management \t\r  "
    assert classifier.clean_text(messy_text) == "feature centralized dotenv management"
    
    # Test complex punctuation and special character stripping
    punctuation_text = "Bug! Error on line #42; crashing... fix instantly? [urgent]"
    assert classifier.clean_text(punctuation_text) == "bug error on line 42 crashing fix instantly urgent"


def test_match_rules_overlapping_keywords():
    classifier = IssueClassifier()
    
    # 'federated' is registered under both 'advanced' architecture and 'ml/ai' domains
    overlap_text = "implement federated model aggregation rules"
    matched_labels = classifier.match_rules(overlap_text)
    
    # Verify that BOTH labels are triggered cleanly from a single keyword token
    assert "level:advanced" in matched_labels
    assert "ml/ai" in matched_labels


def test_predict_fallback_when_sklearn_missing():
    classifier = IssueClassifier()
    
    # Force mock HAS_SKLEARN flag to False to trigger safety defaults
    with patch("src.model.issue_triage.HAS_SKLEARN", False):
        issue_text = "general updates to project documentation assets"
        
        # Verify the classifier falls back to rule matching or defaults without crashing
        prediction = classifier.predict(issue_text, "")
        assert prediction is not None


def test_predict_no_matching_keywords_defaults():
    classifier = IssueClassifier()
    
    # Test text that contains zero registered keywords or domain indicators
    obscure_text = "xyz status value placeholder lookups"
    prediction = classifier.predict(obscure_text, "")
    
    # Verify the system falls back to standard default tracking configurations safely
    assert prediction["type"]["label"] == "bug"
    assert prediction["priority"]["label"] == "medium"


def test_security_keyword_override_takes_precedence():
    classifier = IssueClassifier()
    
    # Text contains regular doc keywords ('README', 'typo') AND a major security keyword ('SQL injection')
    conflicting_text = "Update README documentation typo and fix critical SQL injection vulnerabilities"
    prediction = classifier.predict(conflicting_text, "")
    
    # The safety/security rule override MUST take precedence over the documentation tags
    assert prediction["type"]["label"] == "security"
    assert prediction["priority"]["label"] == "critical"
    assert prediction["level"]["label"] == "critical"


# ===========================================================================
# ORIGINAL WEBHOOK & NLP TRIAGE TESTS
# ===========================================================================

def test_issue_classifier_nlp_and_rules():
    classifier = IssueClassifier()
    
    res_ml = classifier.predict("Build a collaborative filtering recommender system", "")
    assert res_ml["domain"]["label"] == "ml"
    
    res_fe = classifier.predict("CSS styling issues on footer layout alignment", "")
    assert res_fe["domain"]["label"] == "frontend"
    
    res_be = classifier.predict("Create database connection pools for FastAPI backend server", "")
    assert res_be["domain"]["label"] == "backend"


def test_get_suggested_assignees():
    assert "ml-expert-dev" in get_suggested_assignees("ml")
    assert "ui-designer-dev" in get_suggested_assignees("frontend")
    assert "backend-core-dev" in get_suggested_assignees("backend")


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


@pytest.mark.anyio
async def test_triage_issue_skips_api_if_no_token():
    res = await triage_issue(issue_number=100, title="Test issue", body="Frontend button misalignment", repo_full_name="org/repo", token="")
    assert res["issue_number"] == 100
    assert res["github_api"]["status"] == "skipped"


@pytest.mark.anyio
async def test_triage_issue_calls_api_if_token(monkeypatch):
    mock_apply = AsyncMock(return_value={"labels": 200, "comment": 201})
    monkeypatch.setattr("src.model.issue_triage.apply_github_actions", mock_apply)
    
    res = await triage_issue(issue_number=200, title="SQL Injection vulnerability", body="Severe security leak", repo_full_name="org/repo", token="fake-token")
    assert res["github_api"] == {"labels": 200, "comment": 201}


def test_webhook_signature_verification(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "super-secret-key")
    client = TestClient(main.app)
    
    res_no_sig = client.post("/api/webhook/github", json={"action": "opened"})
    assert res_no_sig.status_code == 401
    
    payload = {"action": "opened", "issue": {"number": 1, "title": "t", "body": "b"}, "repository": {"full_name": "o/r"}}
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    sig = hmac.new(b"super-secret-key", payload_bytes, hashlib.sha256).hexdigest()
    headers_valid = {"X-GitHub-Event": "issues", "X-Hub-Signature-256": f"sha256={sig}"}
    
    mock_triage = AsyncMock(return_value={"status": "mocked"})
    monkeypatch.setattr("backend.main.triage_issue", mock_triage)