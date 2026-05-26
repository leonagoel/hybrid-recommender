"""
LLM-based Explanation Generator for Hybrid Recommender System.

Uses Google Generative AI to generate human-readable explanations
for why items were recommended.
"""

import os
import logging
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)
GOOGLE_API_KEY = "Your_API_KEY"

class LLMExplainer:
    """Generate natural language explanations for recommendations using LLM."""

    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None):
        """
        Initialize the LLM explainer.

        Args:
            model_name: Google Generative AI model to use (default: gemini-pro)
            api_key: Google API key (will read from GOOGLE_API_KEY env var if not provided)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or GOOGLE_API_KEY

        if genai is None:
            logger.warning(
                "google-generativeai not installed. Falling back to text-based explanations."
            )
            self.client = None
            return

        if not self.api_key or self.api_key == "Your_API_KEY":
            logger.warning(
                "GOOGLE_API_KEY not found. Set it as an environment variable to enable LLM explanations."
            )
            self.client = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI: {e}")
                self.client = None

    def _build_prompt(
        self,
        recommended_item: str,
        query_item: str,
        scores: Dict[str, float],
        description: str = "",
        top_reviews: list = None,
        category: str = "",
    ) -> str:
        """
        Build a prompt for the LLM to generate an explanation.

        Args:
            recommended_item: The item being recommended
            query_item: The item the user queried
            scores: Dict with keys 'hybrid', 'content', 'collab', 'sentiment'
            description: Item description
            top_reviews: List of top reviews
            category: Item category
            Reason: Why the item is recommended (main reason based on scores)

        Returns:
            Formatted prompt for LLM
        """
        scores_str = "\n".join(
            f"  - {k.capitalize()}: {v:.2%}" for k, v in scores.items() if v is not None
        )

        reviews_str = ""
        if top_reviews:
            reviews_str = "\n\nTop user reviews:\n" + "\n".join(
                f'  - "{review[:100]}..."' for review in top_reviews[:3]
            )

        description_str = f"\nDescription: {description}" if description else ""
        category_str = f"  Category: {category}" if category else ""

        prompt = f"""You are a recommendation system explainer. Generate a comprehensive, engaging explanation 
(3-4 sentences max) for why this item is recommended.

Query item: {query_item}
Recommended item: {recommended_item}{category_str}{description_str}

Recommendation scores:
{scores_str}{reviews_str}

Provide a detailed explanation that:
1. Mentions the MAIN reasons (highest scoring components)
2. Explains WHY it's similar/relevant to what the user is looking for
3. Highlights key features or benefits
4. Is conversational, engaging and non-technical

Generate a COMPLETE, FULL explanation (not truncated):"""
        return prompt

    def explain_recommendation(
        self,
        recommended_item: str,
        query_item: str,
        scores: Dict[str, float],
        description: str = "",
        top_reviews: list = None,
        category: str = "",
    ) -> Optional[str]:
        """
        Generate an LLM-based explanation for a recommendation.

        Args:
            recommended_item: The item being recommended
            query_item: The item the user queried
            scores: Dict with scores (hybrid, content, collab, sentiment)
            description: Item description
            top_reviews: List of top reviews
            category: Item category

        Returns:
            Generated explanation text, or None if LLM is unavailable
        """
        if not self.client:
            # Fallback: Generate a simple text-based explanation
            logger.warning("LLM client not initialized. Generating fallback explanation.")
            return self._generate_fallback_explanation(
                recommended_item, query_item, scores, description, category
            )

        try:
            prompt = self._build_prompt(
                recommended_item=recommended_item,
                query_item=query_item,
                scores=scores,
                description=description,
                top_reviews=top_reviews,
                category=category,
            )

            response = self.client.generate_content(prompt)
            if response is None:
                logger.warning("LLM returned None response, using fallback")
                return self._generate_fallback_explanation(
                    recommended_item, query_item, scores, description, category
                )
            if not hasattr(response, 'text') or response.text is None:
                logger.warning("LLM returned empty response, using fallback")
                return self._generate_fallback_explanation(
                    recommended_item, query_item, scores, description, category
                )
            return response.text.strip()

            response_text = getattr(response, "text", None)
            if response_text:
                return response_text.strip()

            logger.warning("LLM returned empty response, using fallback")
            return self._generate_fallback_explanation(
                recommended_item, query_item, scores, description, category
            )

        except Exception as e:
            logger.error(f"Error generating LLM explanation: {e}. Using fallback explanation.")
            return self._generate_fallback_explanation(
                recommended_item, query_item, scores, description, category
            )
    
    def _generate_fallback_explanation(
        self,
        recommended_item: str,
        query_item: str,
        scores: Dict[str, float],
        description: str = "",
        category: str = "",
    ) -> str:
        """Generate a detailed text-based explanation when LLM is unavailable."""
        # Find the highest scoring component
        valid_scores = {k: v for k, v in scores.items() if v is not None and isinstance(v, (int, float))}
        max_score_name = max(valid_scores, key=valid_scores.get) if valid_scores else "hybrid"
        max_score_value = valid_scores.get(max_score_name, 0) if valid_scores else 0
        
        explanations = {
            "hybrid": f"This {category if category else 'item'} matches your interests across multiple recommendation factors including content similarity, user preferences, and sentiment analysis.",
            "content": f"This {category if category else 'item'} shares similar content features and characteristics with '{query_item}'. Based on content analysis, it has high relevance to your search query.",
            "collab": f"Users who were interested in '{query_item}' also highly appreciated this {category if category else 'item'}. This is based on collaborative filtering across user preferences.",
            "sentiment": f"This {category if category else 'item'} has strong positive reviews and excellent user sentiment scores, indicating high customer satisfaction.",
        }
        
        base_explanation = explanations.get(
            max_score_name, 
            f"This item scored {max_score_value:.1%} match with your search criteria based on hybrid recommendation analysis."
        )
        
        # Add description if available - increased from 100 to 300 characters
        if description and description.strip():
            desc_snippet = description[:300].strip()
            if len(description) > 300:
                desc_snippet += "..."
            return f"{base_explanation} {desc_snippet}"
        return base_explanation

    def explain_multiple(
        self,
        recommendations: list,
        query_item: str,
    ) -> list:
        """
        Generate explanations for multiple recommendations.

        Args:
            recommendations: List of recommendation dicts (must include 'title' and score fields)
            query_item: The item the user queried

        Returns:
            List of recommendations with added 'llm_explanation' field
        """
        results = []
        for rec in recommendations:
            scores = {
                "hybrid": rec.get("hybrid_score"),
                "content": rec.get("content_score"),
                "collab": rec.get("collab_score"),
                "sentiment": rec.get("sentiment_score"),
            }

            explanation = self.explain_recommendation(
                recommended_item=rec.get("title", "Unknown"),
                query_item=query_item,
                scores={k: v for k, v in scores.items() if v is not None},
                description=rec.get("description", ""),
                top_reviews=rec.get("top_reviews", []),
                category=rec.get("category", ""),
            )

            rec_with_explanation = rec.copy()
            rec_with_explanation["llm_explanation"] = explanation
            results.append(rec_with_explanation)

        return results


# Singleton instance for easy use
_explainer_instance: Optional[LLMExplainer] = None


def get_explainer(model_name: str = "gemini-pro") -> LLMExplainer:
    """Get or create a singleton LLMExplainer instance."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = LLMExplainer(model_name=model_name)
    return _explainer_instance
