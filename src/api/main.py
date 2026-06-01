"""
FastAPI Backend for Hybrid Recommender
"""
import os
import sys
from pathlib import Path  # <-- Added
from dotenv import load_dotenv  # <-- Added
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

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
# ===========================================================================
# NEW: Dynamic Configuration Layout Environment Fetching
# ===========================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project-ref.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "your-anon-key-here")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Fetch and clean the comma-separated CORS origins string into a clean list array
RAW_CORS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
CORS_ORIGINS = [origin.strip() for origin in RAW_CORS.split(",")]
# ===========================================================================


class RecommendationRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    top_n: int = 10
    # Set to True to apply IPS causal debiasing on the hybrid score.
    # Downweights items that were over-exposed in training data (popularity/category bias).
    use_causal: bool = False
    # λ blend factor: 0.0 = pure correlation, 1.0 = full IPS reweighting.
    causal_lambda: float = 0.5
    # IPS weight cap — prevents rare items from dominating after reweighting.
    causal_clip: float = 5.0
    fairness: Optional[bool] = None
    fairness_key: Optional[str] = None
    fairness_max_share: Optional[float] = None


# Global read-only model state — never mutated after startup.
_content_model: Optional[ContentRecommender] = None
_collab_model: Optional[CollaborativeRecommender] = None
_item_df = None
<<<<<<< HEAD
fairness: Optional[bool] = None
fairness_key: Optional[str] = None
fairness_max_share: Optional[float] = None
=======
>>>>>>> fea1db0 (fix: safeguard github webhook against null payloads and fix api indentation)


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
        print("Warning: No datasets found for API startup.")
        return

    interaction_df, item_df = dm.merge_all()
    _item_df = item_df
    _content_model = ContentRecommender(item_df)
    if len(interaction_df) > 0 and interaction_df["user_id"].nunique() > 1:
        _collab_model = CollaborativeRecommender(interaction_df)


@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    if _content_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # ===================================================================
    # Try the Primary Hybrid Pipeline
    # ===================================================================
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

<<<<<<< HEAD
        recs = model.recommend(title=req.query, user_id=req.user_id, top_n=req.top_n)
        return {
            "recommendations": recs,
            "causal_debiasing_applied": req.use_causal,
            "fallback": False
        }

    # ===================================================================
    # Graceful Popularity Fallback Recovery Layer (#678)
    # ===================================================================
    except Exception as exc:
        import logging
        logger = logging.getLogger("uvicorn.error")
        logger.error(f"Primary recommendation engine failed: {str(exc)}. Triggering popularity fallback.")
        
        try:
            # Fallback calculation: safe data pull from the global item dataframe
            if '_item_df' in globals() and _item_df is not None and not _item_df.empty:
                # Fall back to picking items safely from your active dataframe asset
                popular_items = _item_df.head(req.top_n)["title"].tolist()
            else:
                # Absolute zero-dependency static default array
                popular_items = ["Top Trending Item A", "Top Trending Item B", "Top Trending Item C"]
            
            # Format the payload items to mimic real recommendation results
            fallback_recs = [
                {
                    "title": item,
                    "hybrid_score": 1.0,
                    "content_score": "—",
                    "collab_score": "—",
                    "sentiment_score": "—",
                    "rating": "5.0",
                    "category": "Trending"
                }
                for item in popular_items
            ]
            
            return {
                "recommendations": fallback_recs,
                "causal_debiasing_applied": False,
                "fallback": True,
                "note": "Primary pipeline encountered an error. Serving trending fallback layout."
            }
            
        except Exception as fallback_exc:
            logger.critical(f"Critical System Outage: Fallback engine failed: {str(fallback_exc)}")
            raise HTTPException(status_code=500, detail="Recommendation engine completely offline.")
=======
    recs = model.recommend(title=req.query, user_id=req.user_id, top_n=req.top_n)
    return {
        "recommendations": recs,
        "causal_debiasing_applied": req.use_causal,
    }
>>>>>>> fea1db0 (fix: safeguard github webhook against null payloads and fix api indentation)
