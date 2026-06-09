"""
validation.py — Centralized Recommendation Validation & Fallback System
========================================================================
Validates recommendation lists, filters invalid/NaN entries, logs warning events,
and triggers cascading fallbacks (popularity -> top-rated -> system default).
"""

import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

def validate_recommendations(
    recommendations,
    fallback_fn=None,
    top_n=10,
    default_fallback_items=None,
    context=None,
    force_padding=True
):
    """
    Validates a list of recommendations, filters out invalid or NaN entries,
    and falls back if the list is empty or has fewer than top_n items.
    
    Parameters:
    - recommendations: raw recommendation list to validate.
    - fallback_fn: callable that returns fallback recommendations: fallback_fn(top_n=top_n).
    - top_n: target number of recommendations to return.
    - default_fallback_items: list of titles to use as default top-rated fallback.
    - context: identifier for logging ('CF', 'hybrid', 'content', etc.).
    - force_padding: if True, pads recommendations up to top_n using cascading fallbacks.
    """
    valid_recs = []
    has_nan_hybrid = False
    has_nan_collab = False
    has_nan_content = False
    
    if isinstance(recommendations, list):
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            
            title = rec.get("title")
            if not title or not isinstance(title, str) or not title.strip():
                continue
                
            # Copy to prevent mutation issues
            cleaned_rec = rec.copy()
            invalid_score = False
            
            for k, v in list(cleaned_rec.items()):
                if k.endswith("_score") or k == "score" or k == "rating" or k == "hybrid_score":
                    if v is None:
                        invalid_score = True
                        if k == "hybrid_score":
                            has_nan_hybrid = True
                        elif k in ("collab_score", "predicted_score"):
                            has_nan_collab = True
                        elif k == "content_score":
                            has_nan_content = True
                        break
                    try:
                        val = float(v)
                        if math.isnan(val) or math.isinf(val):
                            invalid_score = True
                            if k == "hybrid_score":
                                has_nan_hybrid = True
                            elif k in ("collab_score", "predicted_score"):
                                has_nan_collab = True
                            elif k == "content_score":
                                has_nan_content = True
                            break
                        cleaned_rec[k] = val
                    except (ValueError, TypeError):
                        # Keep string placeholders if they are not convertible
                        pass
            
            if invalid_score:
                continue
                
            valid_recs.append(cleaned_rec)
            
    # Structured logging for specific failures
    if has_nan_hybrid:
        logger.warning("Recommendation fallback triggered: NaN hybrid scores")
    elif has_nan_collab:
        logger.warning("Recommendation fallback triggered: NaN collab scores")
    elif has_nan_content:
        logger.warning("Recommendation fallback triggered: NaN content scores")
        
    if not valid_recs:
        if context == "CF":
            logger.warning("Recommendation fallback triggered: empty CF output")
        elif context == "hybrid":
            logger.warning("Recommendation fallback triggered: empty hybrid output")
        else:
            logger.warning(f"Recommendation fallback triggered: empty {context or 'recommendation'} output")

    # Cascading Fallback Resolution Layer
    if force_padding and len(valid_recs) < top_n:
        # A. Use popularity-based recommendations if available (via fallback_fn)
        if fallback_fn is not None:
            try:
                needed = top_n - len(valid_recs)
                fallback_recs = fallback_fn(top_n=top_n)
                if fallback_recs:
                    # Validate fallback recommendations recursively
                    valid_fallbacks = validate_recommendations(
                        recommendations=fallback_recs,
                        fallback_fn=None,
                        top_n=needed,
                        default_fallback_items=None,
                        context=context,
                        force_padding=False
                    )
                    existing_titles = {r["title"].lower().strip() for r in valid_recs}
                    for fr in valid_fallbacks:
                        if fr["title"].lower().strip() not in existing_titles:
                            # Propagate fallback key if not present
                            fr_copy = fr.copy()
                            fr_copy["fallback"] = True
                            valid_recs.append(fr_copy)
                            if len(valid_recs) >= top_n:
                                break
            except Exception as e:
                logger.error(f"Error executing fallback function: {e}")

        # B. If popularity recommendations unavailable, use default top-rated items
        if len(valid_recs) < top_n and default_fallback_items:
            needed = top_n - len(valid_recs)
            existing_titles = {r["title"].lower().strip() for r in valid_recs}
            for item in default_fallback_items:
                if item.lower().strip() not in existing_titles:
                    valid_recs.append({
                        "title": item,
                        "hybrid_score": 0.0,
                        "content_score": 0.0,
                        "collab_score": 0.0,
                        "sentiment_score": 0.0,
                        "rating": 0.0,
                        "category": "Trending",
                        "description": "Default top-rated fallback",
                        "top_reviews": [],
                        "fallback": True
                    })
                    if len(valid_recs) >= top_n:
                        break

        # C. Never return None, D. Always return at least N valid recommendations
        if len(valid_recs) < top_n:
            hardcoded_defaults = ["Top Trending Item A", "Top Trending Item B", "Top Trending Item C"]
            existing_titles = {r["title"].lower().strip() for r in valid_recs}
            for item in hardcoded_defaults:
                if item.lower().strip() not in existing_titles:
                    valid_recs.append({
                        "title": item,
                        "hybrid_score": 0.0,
                        "content_score": 0.0,
                        "collab_score": 0.0,
                        "sentiment_score": 0.0,
                        "rating": 0.0,
                        "category": "Trending",
                        "description": "Static system fallback",
                        "top_reviews": [],
                        "fallback": True
                    })
                    if len(valid_recs) >= top_n:
                        break
                        
        # If still not enough, repeat default items to satisfy N
        while len(valid_recs) < top_n:
            valid_recs.append({
                "title": f"Default Fallback Item {len(valid_recs) + 1}",
                "hybrid_score": 0.0,
                "content_score": 0.0,
                "collab_score": 0.0,
                "sentiment_score": 0.0,
                "rating": 0.0,
                "category": "Trending",
                "description": "Static fallback spacer",
                "top_reviews": [],
                "fallback": True
            })

    return valid_recs[:top_n] if force_padding else valid_recs
