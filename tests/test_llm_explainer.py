"""
Unit tests for the LLM Explainer module.
Tests both LLM explanations and fallback explanations.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm_explainer import LLMExplainer, get_explainer


class TestLLMExplainerInit:
    """Test LLMExplainer initialization."""

    def test_explainer_initialization(self):
        """Test that explainer initializes without errors."""
        explainer = LLMExplainer(model_name="gemini-pro")
        assert explainer is not None
        assert explainer.model_name == "gemini-pro"

    def test_singleton_pattern(self):
        """Test that get_explainer() returns singleton instance."""
        explainer1 = get_explainer()
        explainer2 = get_explainer()
        assert explainer1 is explainer2


class TestFallbackExplanations:
    """Test fallback explanation generation."""

    def test_fallback_content_explanation(self):
        """Test fallback explanation for content-based recommendation."""
        explainer = LLMExplainer()
        explanation = explainer._generate_fallback_explanation(
            recommended_item="Harry Potter Book",
            query_item="Lord of the Rings",
            scores={"content": 0.85, "hybrid": 0.75, "collab": 0.5},
            description="A magical fantasy novel",
            category="Books"
        )
        assert explanation is not None
        assert len(explanation) > 0
        assert "similar content features" in explanation.lower()

    def test_fallback_collaborative_explanation(self):
        """Test fallback explanation for collaborative recommendation."""
        explainer = LLMExplainer()
        explanation = explainer._generate_fallback_explanation(
            recommended_item="Product A",
            query_item="Product B",
            scores={"collab": 0.95, "hybrid": 0.80},
            category="Electronics"
        )
        assert explanation is not None
        assert "users who" in explanation.lower() or "collaborative" in explanation.lower()

    def test_fallback_sentiment_explanation(self):
        """Test fallback explanation for sentiment-based recommendation."""
        explainer = LLMExplainer()
        explanation = explainer._generate_fallback_explanation(
            recommended_item="Highly Rated Item",
            query_item="Query Item",
            scores={"sentiment": 0.92, "hybrid": 0.70},
            category="Products"
        )
        assert explanation is not None
        assert "positive" in explanation.lower() or "review" in explanation.lower()

    def test_fallback_with_long_description(self):
        """Test that long descriptions are truncated at 300 chars."""
        explainer = LLMExplainer()
        long_desc = "A" * 500  # 500 character description
        explanation = explainer._generate_fallback_explanation(
            recommended_item="Item",
            query_item="Query",
            scores={"hybrid": 0.8},
            description=long_desc,
            category="Test"
        )
        assert explanation is not None
        assert len(explanation) < 400  # Should be truncated

    def test_fallback_empty_scores(self):
        """Test fallback with empty scores dictionary."""
        explainer = LLMExplainer()
        explanation = explainer._generate_fallback_explanation(
            recommended_item="Item A",
            query_item="Item B",
            scores={},
            category="Test"
        )
        assert explanation is not None
        assert len(explanation) > 0


class TestExplainRecommendation:
    """Test single recommendation explanation."""

    def test_explain_recommendation_basic(self):
        """Test basic recommendation explanation."""
        explainer = LLMExplainer()
        explanation = explainer.explain_recommendation(
            recommended_item="Test Item A",
            query_item="Test Item B",
            scores={"hybrid": 0.85, "content": 0.90, "sentiment": 0.75},
            description="A test product",
            category="Test Category"
        )
        # Should return either LLM or fallback explanation
        assert explanation is None or isinstance(explanation, str)
        if explanation:
            assert len(explanation) > 0

    def test_explain_recommendation_multiple_scores(self):
        """Test explanation with multiple scoring components."""
        explainer = LLMExplainer()
        explanation = explainer.explain_recommendation(
            recommended_item="Product A",
            query_item="Product B",
            scores={
                "hybrid": 0.88,
                "content": 0.92,
                "collab": 0.82,
                "sentiment": 0.79
            },
            description="Premium electronics",
            top_reviews=["Great product!", "Highly recommend"],
            category="Electronics"
        )
        assert explanation is None or isinstance(explanation, str)


class TestExplainMultiple:
    """Test batch explanation generation."""

    def test_explain_multiple_recommendations(self):
        """Test explaining multiple recommendations at once."""
        explainer = LLMExplainer()
        recommendations = [
            {
                "title": "Item 1",
                "hybrid_score": 0.92,
                "content_score": 0.88,
                "collab_score": 0.95,
                "sentiment_score": 0.85,
                "description": "First item",
                "category": "Category A"
            },
            {
                "title": "Item 2",
                "hybrid_score": 0.85,
                "content_score": 0.80,
                "collab_score": 0.88,
                "sentiment_score": 0.82,
                "description": "Second item",
                "category": "Category B"
            }
        ]
        
        results = explainer.explain_multiple(recommendations, "Query Item")
        
        assert len(results) == 2
        assert all("llm_explanation" in r for r in results)
        assert all(r.get("llm_explanation") is None or isinstance(r["llm_explanation"], str) for r in results)

    def test_explain_multiple_preserves_data(self):
        """Test that explain_multiple preserves original data."""
        explainer = LLMExplainer()
        original_recs = [
            {
                "title": "Item A",
                "hybrid_score": 0.90,
                "rating": 4.5
            }
        ]
        
        results = explainer.explain_multiple(original_recs, "Query")
        
        # Original fields should still exist
        assert results[0]["title"] == "Item A"
        assert results[0]["hybrid_score"] == 0.90
        assert results[0]["rating"] == 4.5
        assert "llm_explanation" in results[0]


class TestPromptBuilding:
    """Test prompt building for LLM."""

    def test_build_prompt_structure(self):
        """Test that prompts are properly formatted."""
        explainer = LLMExplainer()
        prompt = explainer._build_prompt(
            recommended_item="Book A",
            query_item="Book B",
            scores={"hybrid": 0.85, "content": 0.90},
            description="A fiction novel",
            top_reviews=["Amazing!", "Loved it"],
            category="Books"
        )
        
        assert prompt is not None
        assert "Book A" in prompt
        assert "Book B" in prompt
        assert "Books" in prompt
        assert "Hybrid" in prompt or "hybrid" in prompt

    def test_build_prompt_with_missing_data(self):
        """Test prompt building with minimal data."""
        explainer = LLMExplainer()
        prompt = explainer._build_prompt(
            recommended_item="Item",
            query_item="Query",
            scores={"hybrid": 0.5},
            description="",
            top_reviews=[],
            category=""
        )
        
        assert prompt is not None
        assert len(prompt) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
