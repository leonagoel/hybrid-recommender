"""
FastAPI Backend for Hybrid Recommender
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset_manager import DatasetManager
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender

app = FastAPI(title="Hybrid Recommender API")

class RecommendationRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    top_n: int = 10

content_model = None
collab_model = None
hybrid_model = None

@app.on_event("startup")
def startup_event():
    global content_model, collab_model, hybrid_model
    dm = DatasetManager()
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets')
    
    datasets_to_load = ['books.csv', 'booksdata.csv', 'ratings.csv']
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
    
    content_model = ContentRecommender(item_df)
    if len(interaction_df) > 0 and interaction_df['user_id'].nunique() > 1:
        collab_model = CollaborativeRecommender(interaction_df)
    
    hybrid_model = HybridRecommender(content_model, collab_model, item_df)

@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    if not hybrid_model:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    recs = hybrid_model.recommend(title=req.query, user_id=req.user_id, top_n=req.top_n)
    return {"recommendations": recs}
