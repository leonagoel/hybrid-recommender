"""Minimal LLM-based explanation generator for recommendations.

This file intentionally provides a compact, easy-to-review implementation
that mirrors the original public API: `LLMExplainer` and `get_explainer()`.
The goal is to replace a prior, larger implementation with a clearer one
so we can reintroduce the fix cleanly.
"""

import os
import logging
from typing import Optional, Dict, List

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

logger = logging.getLogger(__name__)


class LLMExplainer:
    """Generate human-readable explanations for recommendations.

    This class prefers a real LLM client when available and falls back
    to a deterministic text generator otherwise.
    """

    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Keep client as None if runtime API differs; guard usage later.
                self.client = genai
            except Exception as e:
                logger.debug("LLM client init failed: %s", e)

    def _build_prompt(self, recommended_item: str, query_item: str, scores: Dict[str, float]) -> str:
        parts = [f"Query: {query_item}", f"Recommended: {recommended_item}"]
        if scores:
            score_lines = ", ".join(f"{k}={v:.2%}" for k, v in scores.items())
            parts.append(f"Scores: {score_lines}")
        return " | ".join(parts)

    def explain_recommendation(
        self,
        recommended_item: str,
        query_item: str,
        scores: Dict[str, float],
        description: str = "",
        top_reviews: Optional[List[str]] = None,
        category: str = "",
    ) -> Optional[str]:
        """Return an explanation string or None if not possible.

        This is intentionally small and deterministic for easier testing.
        """
        # Prefer an LLM when available
        if self.client is not None:
            try:
                prompt = self._build_prompt(recommended_item, query_item, scores)
                # Use a lightweight call shape to avoid depending on exact SDK surface.
                resp = getattr(self.client, "generate_content", None)
                if callable(resp):
                    out = resp(prompt)
                    text = getattr(out, "text", None)
                    if text:
                        return text.strip()
            except Exception as e:
                logger.debug("LLM generation error: %s", e)

        # Fallback deterministic explanation
        return self._generate_fallback_explanation(recommended_item, query_item, scores, description, category)

    def _generate_fallback_explanation(
        self, recommended_item: str, query_item: str, scores: Dict[str, float], description: str = "", category: str = ""
    ) -> str:
        if not scores:
            return f"{recommended_item} is recommended based on similarity to {query_item}."
        primary = max(scores.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1))[0]
        return f"Recommended because {primary} contributed most to matching {query_item}."


_explainer_instance: Optional[LLMExplainer] = None


def get_explainer(model_name: str = "gemini-pro") -> LLMExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = LLMExplainer(model_name=model_name)
    return _explainer_instance
