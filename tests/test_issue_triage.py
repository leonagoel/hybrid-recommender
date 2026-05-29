"""
Unit and integration tests for the NLP issue triage classifier, rule overrides,
assignee suggestor, and GitHub API action layouts for Issue #625.
"""

import pytest
import hmac
import hashlib
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Import from your exact structural app paths
from src.model.issue_triage import (
    IssueClassifier,
    get_suggested_assignees,
    format_triage_comment,
    triage_issue,
    apply_github_actions
)


# ===========================================================================
# REQUIRED EDGE-CASE TESTS FOR ISSUE #625
# ===========================================================================

def test_issue_classifier_clean_text_edge_cases():
    """Verifies clean_text handles extreme whitespaces, line breaks, and characters."""
    classifier = IssueClassifier()
    
    # Requirement: Test extreme whitespaces, line breaks, tabs, and carriage returns
    messy_text = "   \n\n  FEatURE:  centralized   dotenv    management \t\r  "
    cleaned = classifier.clean_text(messy_text, "")
    assert cleaned == "feature centralized dotenv management"
    
    # Requirement: Test stripping out complex special punctuation character boundaries
    punctuation_text = "Bug! Error on line #42; crashing... fix instantly? [urgent]"
    cleaned_punct = classifier.clean_text(punctuation_text, "")
    assert cleaned_punct == "bug error on line 42 crashing fix instantly urgent"


def test_match_rules_overlapping_keywords():
    """Verifies keywords like 'federated' match multiple categories cleanly."""
    classifier = IssueClassifier()
    
    # 'federated' is a keyword in both 'ml_keywords' (domain: ml) and 'advanced_keywords' (level: advanced)
    overlap_text = "implement federated model aggregation rules"
    rules_matched = classifier.match_rules(overlap_text)
    
    # Assert that overlapping tokens trigger both structural rule mappings
    assert "domain" in rules_matched and rules_matched["domain"][0] == "ml"
    assert "level" in rules_matched and rules_matched["level"][0] == "advanced"


def test_predict_fallback_when_sklearn_missing():
    """Verifies predict function handles execution safely when scikit-learn is missing."""
    classifier = IssueClassifier()
    
    # Requirement: Force mock HAS_SKLEARN to False to test code path safety
    with patch("src.model.issue_triage.HAS_SKLEARN", False):
        issue_text = "general updates to project documentation assets"
        prediction = classifier.predict(issue_text, "")
        
        # Verify the classifier still computes a standard dict object instead of throwing an unhandled exception
        assert isinstance(prediction, dict)
        assert "type" in prediction
        assert prediction["type"]["reason"] == "Default fallback framework" or "keyword" in prediction["type"]["reason"].lower()


def test_predict_no_matching_keywords_defaults():
    """Verifies fallback values trigger when an issue matches zero keyword tokens."""
    classifier = IssueClassifier()
    
    # Text completely devoid of any seed keywords or category rules
    obscure_text = "xyz status value placeholder lookups"
    
    # Temporarily patch out sklearn to isolate the pure rule-mismatch default block
    with patch("src.model.issue_triage.HAS_SKLEARN", False):
        prediction = classifier.predict(obscure_text, "")
        
        # Assert default fallback schema states apply safely
        assert prediction["type"]["label"] == "bug"
        assert prediction["priority"]["label"] == "medium"
        assert prediction["level"]["label"] == "beginner"


def test_security_keyword_override_takes_precedence():
    """Verifies security indicators take absolute override priority over standard rules."""
    classifier = IssueClassifier()
    
    # Contains a clear beginner/doc keyword ('readme') but also a high-priority security risk ('sql injection')
    conflicting_text = "update the layout readme and fix a severe query sql injection breach vulnerability"
    prediction = classifier.predict(conflicting_text, "")
    
    # Requirement: Assert security takes precedence over the low priority beginner labels
    assert prediction["type"]["label"] == "security"
    assert prediction["priority"]["label"] == "critical"
    assert prediction["level"]["label"] == "critical"


# ===========================================================================
# STRUCTURAL VALIDATION & PAYLOAD TESTS
# ===========================================================================

def test_get_suggested_assignees():
    """Validates assignee lists map cleanly to area profiles."""
    assert "ml-expert-dev" in get_suggested_assignees("ml")
    assert "ui-designer-dev" in get_suggested_assignees("frontend")
    assert "backend-core-dev" in get_suggested_assignees("backend")
    assert get_suggested_assignees("unknown_domain") == []


def test_format_triage_comment():
    """Validates the markdown formatting engine constructs a valid report layout."""
    mock_predictions = {
        "type": {"label": "bug", "confidence": 0.95, "reason": "Test reason"},
        "domain": {"label": "frontend", "confidence": 0.85, "reason": "Test reason"},
        "level": {"label": "beginner", "confidence": 0.75, "reason": "Test reason"},
        "priority": {"label": "low", "confidence": 0.65, "reason": "Test reason"},
    }
    comment = format_triage_comment(mock_predictions, ["ui-designer-dev"])
    
    assert "### 📌 GSSoC 2026 - Issue Auto-Triaged" in comment
    assert "type:bug" in comment
    assert "@ui-designer-dev" in comment


@pytest.mark.anyio
async def test_triage_issue_skips_api_if_no_token():
    """Verifies that webhook processing flows run closed if GITHUB_TOKEN is missing."""
    res = await triage_issue(
        issue_number=101,
        title="CSS button offset defect",
        body="Layout is broken",
        repo_full_name="leonagoel/hybrid-recommender",
        token=""
    )
    assert res["issue_number"] == 101
    assert res["github_api"]["status"] == "skipped"


@pytest.mark.anyio
async def test_triage_issue_executes_api_with_token(monkeypatch):
    """Verifies that actions pipeline triggers calls to the GitHub API endpoints when authorized."""
    mock_actions = AsyncMock(return_value={"labels": 200, "comment": 201})
    monkeypatch.setattr("src.model.issue_triage.apply_github_actions", mock_actions)
    
    res = await triage_issue(
        issue_number=490,
        title="Critical data leakage threat detected",
        body="Secret token visible in text dump logs",
        repo_full_name="leonagoel/hybrid-recommender",
        token="ghp_mockValidToken"
    )
    
    assert res["github_api"]["labels"] == 200
    assert res["github_api"]["comment"] == 201