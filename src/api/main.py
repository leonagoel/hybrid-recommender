"""
FastAPI Backend for Hybrid Recommender
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Calculate absolute paths and load environment variables first
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent  # Steps out of src/api to project root

ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Fix the path mapping so internal src imports work perfectly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset_manager import DatasetManager
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender
from src.model.causal_config import CausalConfig

app = FastAPI(title="Hybrid Recommender API")


class RecommendationRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    top_n: int = 10

    # Apply IPS causal debiasing on the hybrid score.
    use_causal: bool = False
    causal_lambda: float = 0.5
    causal_clip: float = 5.0

    fairness: Optional[bool] = None
    fairness_key: Optional[str] = None
    fairness_max_share: Optional[float] = None


# Global read-only model state — never mutated after startup.
_content_model: Optional[ContentRecommender] = None
_collab_model: Optional[CollaborativeRecommender] = None
_item_df = None


def _make_trending_fallback(req: RecommendationRequest) -> dict:
    # Best-effort safe pull from the global item dataframe.
    fallback_titles = []
    if _item_df is not None and not _item_df.empty and "title" in _item_df.columns:
        fallback_titles = _item_df.head(max(0, req.top_n))[["title"]].copy()
        fallback_titles = fallback_titles["title"].astype(str).tolist()

    if not fallback_titles:
        fallback_titles = [
            "Top Trending Item A",
            "Top Trending Item B",
            "Top Trending Item C",
        ][: max(1, req.top_n)]

    fallback_recs = [
        {
            "title": item,
            "hybrid_score": 1.0,
            "content_score": "—",
            "collab_score": "—",
            "sentiment_score": "—",
            "rating": "5.0",
            "category": "Trending",
        }
        for item in fallback_titles[: max(0, req.top_n)]
    ]

    return {
        "recommendations": fallback_recs,
        "model_name": "hybrid",
        "message": "Models not loaded. Serving trending fallback layout.",
        "causal_debiasing_applied": False,
        "fallback": True,
        "note": "Models not loaded. Serving trending fallback layout.",
    }


@app.on_event("startup")
def startup_event():
    global _content_model, _collab_model, _item_df

    dm = DatasetManager()
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets"
    )

    datasets_to_load = ["books.csv", "booksdata.csv", "ratings.csv"]
    loaded = False

    for filename in datasets_to_load:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            dm.load_csv(filepath)
            loaded = True
            break

    if not loaded:
        # Keep app running; /recommend will serve fallback.
        return

    interaction_df, item_df = dm.merge_all()
    _item_df = item_df
    _content_model = ContentRecommender(item_df)

    if len(interaction_df) > 0 and interaction_df["user_id"].nunique() > 1:
        _collab_model = CollaborativeRecommender(interaction_df)


@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    # If models haven't loaded (startup missed datasets), serve fallback instead.
    if _content_model is None:
        return _make_trending_fallback(req)

    # Try the Primary Hybrid Pipeline
    try:
        causal_cfg = (
            CausalConfig(
                enabled=True,
                blend_lambda=req.causal_lambda,
                clip_max=req.causal_clip,
            )
            if req.use_causal
            else CausalConfig.disabled()
        )

        model = HybridRecommender(
            _content_model,
            _collab_model,
            _item_df,
            causal_config=causal_cfg,
        )

        recs = model.recommend(
            title=req.query,
            user_id=req.user_id,
            top_n=req.top_n,
        )

        return {
            "recommendations": recs,
            "model_name": "hybrid",
            "message": "Recommendations retrieved successfully",
            "causal_debiasing_applied": req.use_causal,
            "fallback": False,
        }

    # Graceful Popularity Fallback Recovery Layer (#678)
    except Exception as exc:
        # Absolute last-resort: never leak exception details to the client.
        try:
            payload = _make_trending_fallback(req)
            payload["message"] = "Primary pipeline encountered an error. Serving trending fallback layout."
            payload["note"] = "Primary pipeline encountered an error. Serving trending fallback layout."
            payload["causal_debiasing_applied"] = False
            payload["fallback"] = True
            return payload
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Recommendation engine completely offline.",
            )


@app.post("/recommendations")
def get_recommendations_legacy(req: RecommendationRequest):
    """
    Backward-compatible alias for clients and issue reports that call /recommendations.
    """
    return get_recommendations(req)
