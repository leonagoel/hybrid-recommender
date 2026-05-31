"""
Issue Triage System using NLP and Regex Rules.
Classifies incoming issues by Type, Domain, Level, and Priority,
recommends assignees based on domain expertise, and handles GitHub API actions.
"""

import os
import re
import logging
import httpx
from typing import Dict, List, Tuple, Any
import numpy as np

# Ensure sklearn is installed
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# Domain expertise assignee mapping
DOMAIN_EXPERTISE_MAP = {
    "ml": ["ml-expert-dev", "nlp-researcher"],
    "frontend": ["ui-designer-dev", "react-expert"],
    "backend": ["backend-core-dev", "db-admin"]
}

# Seed dataset for dynamic training
SEED_DATA = [
    # Bugs
    {"text": "Null pointer exception in database query parsing and execution", "type": "bug", "domain": "backend", "level": "beginner", "priority": "medium"},
    {"text": "FastAPI endpoint returns 500 server error when JSON payload is empty", "type": "bug", "domain": "backend", "level": "beginner", "priority": "high"},
    {"text": "Database connection timeout during concurrent query execution", "type": "bug", "domain": "backend", "level": "advanced", "priority": "medium"},
    {"text": "Syntax error in SQL migration script for products table schema", "type": "bug", "domain": "backend", "level": "beginner", "priority": "low"},
    {"text": "REST API fails to parse query parameters with trailing spaces", "type": "bug", "domain": "backend", "level": "beginner", "priority": "low"},
    {"text": "Footer overlap on small screen mobile devices and layouts", "type": "bug", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "CSS color variable not applied correctly on button hover state", "type": "bug", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Search box border alignment offset by 2px in Safari browser", "type": "bug", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "React compilation warning: unused imports in navigation bar component", "type": "bug", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Dropdown menu items cut off in dashboard UI grid layout", "type": "bug", "domain": "frontend", "level": "beginner", "priority": "medium"},
    {"text": "SVD collaborative model crashes with ValueError: singular matrix in decomposition", "type": "bug", "domain": "ml", "level": "advanced", "priority": "high"},
    {"text": "TfidfVectorizer throws out of memory exception on large corpus datasets", "type": "bug", "domain": "ml", "level": "advanced", "priority": "high"},
    {"text": "Bayesian ranking rating returns NaN when product vote count is zero", "type": "bug", "domain": "ml", "level": "beginner", "priority": "medium"},
    {"text": "Collaborative model predictions mismatch during evaluation cycle runs", "type": "bug", "domain": "ml", "level": "advanced", "priority": "medium"},
    {"text": "Sentiment analysis analyzer fails to load nltk VADER lexicon files", "type": "bug", "domain": "ml", "level": "beginner", "priority": "medium"},
    
    # Features
    {"text": "Add federated learning training support to collaborative models", "type": "feature", "domain": "ml", "level": "advanced", "priority": "high"},
    {"text": "Implement cross-domain recommendation matching for multi-catalog data", "type": "feature", "domain": "ml", "level": "advanced", "priority": "high"},
    {"text": "Build deep learning transformer-based item description encoder pipeline", "type": "feature", "domain": "ml", "level": "advanced", "priority": "medium"},
    {"text": "Introduce hybrid sentiment weighting factor for recommendation scores", "type": "feature", "domain": "ml", "level": "advanced", "priority": "medium"},
    {"text": "Train lightweight BERT classifier for auto-labeling text descriptions", "type": "feature", "domain": "ml", "level": "advanced", "priority": "medium"},
    {"text": "Create responsive navigation header with dark mode toggle switch", "type": "feature", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Add product category filtration badges below the main search input", "type": "feature", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Implement interactive rating slider component for feedback page UI", "type": "feature", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Build loading skeleton grids to show when products are being fetched", "type": "feature", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Customize theme variables for dynamic layout color changing", "type": "feature", "domain": "frontend", "level": "beginner", "priority": "low"},
    {"text": "Expose GET api status health check API route", "type": "feature", "domain": "backend", "level": "beginner", "priority": "low"},
    {"text": "Add endpoint to export current catalog items to a CSV file download", "type": "feature", "domain": "backend", "level": "beginner", "priority": "low"},
    {"text": "Implement API response caching layer using Redis backend server", "type": "feature", "domain": "backend", "level": "advanced", "priority": "medium"},
    {"text": "Add rate limiting headers to prevent abuse on search endpoints", "type": "feature", "domain": "backend", "level": "beginner", "priority": "medium"},
    {"text": "Create purchase tracking POST API route to record user interactions", "type": "feature", "domain": "backend", "level": "beginner", "priority": "low"},
    
    # Security
    {"text": "Migrate password hashing function from MD5 to bcrypt with cost factor 12", "type": "security", "domain": "backend", "level": "advanced", "priority": "critical"},
    {"text": "Prevent SQL injection in search RPC by using parameterized query binds", "type": "security", "domain": "backend", "level": "advanced", "priority": "critical"},
    {"text": "Expose credentials security leak in public git history files credentials token", "type": "security", "domain": "backend", "level": "advanced", "priority": "critical"},
    {"text": "Add JWT token validation middleware to protect sensitive admin APIs auth", "type": "security", "domain": "backend", "level": "advanced", "priority": "high"},
    {"text": "Fix Cross-Site Scripting vulnerability in markdown review comments parser XSS", "type": "security", "domain": "backend", "level": "advanced", "priority": "high"},
    {"text": "Sanitize user inputs to prevent XSS in description rendering elements frontend HTML", "type": "security", "domain": "frontend", "level": "advanced", "priority": "high"},
    {"text": "Configure Content Security Policy CSP headers to block unsafe script injection", "type": "security", "domain": "frontend", "level": "advanced", "priority": "high"},
    {"text": "Harden frontend authentication flow and secure cookies against CSRF attacks", "type": "security", "domain": "frontend", "level": "advanced", "priority": "high"},
]


class IssueClassifier:
    """
    NLP and Rule-based issue classifier.
    Trains on SEED_DATA using TF-IDF + Logistic Regression and uses regex checks.
    """

    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.categories = ["type", "domain", "level", "priority"]
        
        if HAS_SKLEARN:
            self._train()
            
    def _train(self):
        texts = [item["text"].lower() for item in SEED_DATA]
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        
        for category in self.categories:
            y = [item[category] for item in SEED_DATA]
            clf = LogisticRegression(C=5.0, class_weight="balanced", random_state=42)
            clf.fit(X, y)
            self.models[category] = clf

    def clean_text(self, title: str, body: str) -> str:
        text = f"{title} {body or ''}"
        text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text)
        return " ".join(text.lower().split())

    def match_rules(self, text: str) -> Dict[str, Tuple[str, float, str]]:
        """
        Applies deterministic keyword and regex rules.
        Returns: Dict containing updates to predictions {category: (label, confidence, reason)}
        """
        rules = {}
        
        # Security overrides
        sec_keywords = ["security", "vulnerability", "leak", "secret", "token", "password", "md5", "bcrypt", "csrf", "xss", "auth", "encrypt", "cve"]
        if any(kw in text for kw in sec_keywords):
            rules["type"] = ("security", 1.0, f"Matched security keyword(s): {[kw for kw in sec_keywords if kw in text]}")
            rules["level"] = ("critical", 1.0, "Security issue labeled as critical level")
            rules["priority"] = ("critical", 1.0, "Security issue escalated to critical priority")

        # ML indicators
        ml_keywords = ["ml", "ai", "svd", "collaborative", "recommender", "model", "train", "dataset", "embeddings", "sentiment", "nlp", "vader"]
        if any(kw in text for kw in ml_keywords):
            rules["domain"] = ("ml", 0.95, f"Matched ML keyword(s): {[kw for kw in ml_keywords if kw in text]}")

        # Frontend indicators
        fe_keywords = ["css", "html", "react", "frontend", "ui", "view", "component", "button", "align", "color", "styling", "theme", "mobile", "layout", "grid", "responsive"]
        if any(kw in text for kw in fe_keywords) and "domain" not in rules:
            rules["domain"] = ("frontend", 0.95, f"Matched frontend keyword(s): {[kw for kw in fe_keywords if kw in text]}")

        # Backend indicators
        be_keywords = ["api", "endpoint", "backend", "fastapi", "route", "database", "supabase", "server", "postgres", "sql", "webhook", "redis", "caching"]
        if any(kw in text for kw in be_keywords) and "domain" not in rules:
            rules["domain"] = ("backend", 0.95, f"Matched backend keyword(s): {[kw for kw in be_keywords if kw in text]}")

        # Difficulty Level indicators
        beginner_keywords = ["beginner", "easy", "simple", "typo", "doc", "documentation", "readme", "good first issue"]
        if any(kw in text for kw in beginner_keywords) and "level" not in rules:
            rules["level"] = ("beginner", 0.9, f"Matched beginner keyword(s): {[kw for kw in beginner_keywords if kw in text]}")
            if "priority" not in rules:
                rules["priority"] = ("low", 0.9, "Beginner tasks marked as low priority")
                
        advanced_keywords = ["advanced", "performance", "architecture", "federated", "concurrency", "race condition", "multithreading", "optimizing", "memory leak"]
        if any(kw in text for kw in advanced_keywords) and "level" not in rules:
            rules["level"] = ("advanced", 0.9, f"Matched advanced keyword(s): {[kw for kw in advanced_keywords if kw in text]}")
            if "priority" not in rules:
                rules["priority"] = ("high", 0.9, "Advanced tasks marked as high priority")

        # Priority indicators
        high_priority_keywords = ["urgent", "critical", "broken", "severe", "crash", "error 500", "fails to build"]
        if any(kw in text for kw in high_priority_keywords):
            if "priority" not in rules or rules["priority"][0] != "critical":
                rules["priority"] = ("high", 0.95, f"Matched high-priority keyword(s): {[kw for kw in high_priority_keywords if kw in text]}")

        return rules

    def predict(self, title: str, body: str) -> Dict[str, Dict[str, Any]]:
        """
        Classifies the issue using NLP (if available) and Regex rules.
        Returns:
            Dict of {category: {"label": str, "confidence": float, "reason": str}}
        """
        text = self.clean_text(title, body)
        predictions = {}
        
        # NLP classifier prediction
        if HAS_SKLEARN and self.vectorizer is not None:
            X = self.vectorizer.transform([text])
            for cat in self.categories:
                clf = self.models[cat]
                probs = clf.predict_proba(X)[0]
                classes = clf.classes_
                idx = np.argmax(probs)
                predictions[cat] = {
                    "label": str(classes[idx]),
                    "confidence": float(probs[idx]),
                    "reason": "NLP classification"
                }
        else:
            # Empty default predictions if no classifier is trained
            default_labels = {
                "type": "bug",
                "domain": "backend",
                "level": "beginner",
                "priority": "medium"
            }
            for cat in self.categories:
                predictions[cat] = {
                    "label": default_labels[cat],
                    "confidence": 0.5,
                    "reason": "Default fallback"
                }

        # Rule overrides
        rules = self.match_rules(text)
        for cat, (label, conf, reason) in rules.items():
            predictions[cat] = {
                "label": label,
                "confidence": conf,
                "reason": reason
            }

        return predictions


def get_suggested_assignees(domain: str) -> List[str]:
    """Returns developer handles based on domain expertise mapping."""
    return DOMAIN_EXPERTISE_MAP.get(domain.lower(), [])


def format_triage_comment(predictions: Dict[str, Dict[str, Any]], assignees: List[str]) -> str:
    """Formats a user-friendly GSSoC triage comment with reasoning."""
    def clean_label(cat: str, val: str) -> str:
        if cat == "type":
            return f"type:{val}"
        if cat == "domain":
            return "ml/ai" if val == "ml" else val
        if cat == "level":
            return f"level:{val}"
        if cat == "priority":
            return f"priority:{val}"
        return val

    rows = []
    for cat in ["type", "domain", "level", "priority"]:
        pred = predictions[cat]
        label = clean_label(cat, pred["label"])
        rows.append(f"| **{cat.capitalize()}** | `{label}` | {pred['reason']} (conf: {pred['confidence']:.2f}) |")

    assignee_str = ", ".join([f"@{a}" for a in assignees]) if assignees else "None suggested"
    
    comment = f"""### 📌 GSSoC 2026 - Issue Auto-Triaged

| Category | Detected Label | Reasoning |
| --- | --- | --- |
{chr(10).join(rows)}

**Suggested Assignee(s)**: {assignee_str}
*(Based on area expertise mapping)*

---
*Note: A human moderator can always override these choices. Add a comment containing `!retriage` to re-run the triage classifier.*
"""
    return comment.strip()


async def apply_github_actions(
    repo_full_name: str,
    issue_number: int,
    predictions: Dict[str, Dict[str, Any]],
    assignees: List[str],
    token: str
) -> Dict[str, Any]:
    """
    Interacts with GitHub API to:
    1. Apply labels corresponding to classification.
    2. Suggest/Add assignees if permitted, or post suggested assignees in the comment.
    3. Post the triage analysis reasoning comment.
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Map raw predictions to GitHub label format
    labels = []
    # 1. Type
    t_label = predictions["type"]["label"]
    labels.append(f"type:{t_label}")
    # 2. Domain
    d_label = predictions["domain"]["label"]
    labels.append("ml/ai" if d_label == "ml" else d_label)
    # 3. Level
    l_label = predictions["level"]["label"]
    labels.append(f"level:{l_label}")
    
    # We also add 'gssoc:approved' by default since this is GSSoC auto-triage
    labels.append("gssoc:approved")

    api_url = f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}"
    
    results = {}
    
    # Add labels
    async with httpx.AsyncClient() as client:
        try:
            label_res = await client.post(
                f"{api_url}/labels",
                json={"labels": labels},
                headers=headers
            )
            results["labels"] = label_res.status_code
        except Exception as e:
            logger.error("GitHub API labels application failed: %s", e)
            results["labels"] = f"error: {str(e)}"
            
        # Post comment
        comment_body = format_triage_comment(predictions, assignees)
        try:
            comment_res = await client.post(
                f"{api_url}/comments",
                json={"body": comment_body},
                headers=headers
            )
            results["comment"] = comment_res.status_code
        except Exception as e:
            logger.error("GitHub API comment posting failed: %s", e)
            results["comment"] = f"error: {str(e)}"

    return results


async def triage_issue(
    issue_number: int,
    title: str,
    body: str,
    repo_full_name: str,
    token: str = ""
) -> Dict[str, Any]:
    """Main function called by webhook to process issue and apply actions."""
    classifier = IssueClassifier()
    predictions = classifier.predict(title, body)
    
    domain = predictions["domain"]["label"]
    assignees = get_suggested_assignees(domain)
    
    triage_info = {
        "issue_number": issue_number,
        "predictions": predictions,
        "suggested_assignees": assignees,
    }
    
    if token:
        github_results = await apply_github_actions(
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            predictions=predictions,
            assignees=assignees,
            token=token
        )
        triage_info["github_api"] = github_results
    else:
        logger.info("Skipping GitHub API actions (GITHUB_TOKEN not configured).")
        triage_info["github_api"] = {"status": "skipped"}
        
    return triage_info
