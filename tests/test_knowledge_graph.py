"""
Unit and integration tests for KnowledgeGraphRecommender, TransE embeddings training,
HybridRecommender integration, and FastAPI build API routing.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.model.knowledge_graph import KnowledgeGraphRecommender
from src.model.hybrid_model import HybridRecommender
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from backend import main


def test_knowledge_graph_triples_and_indexing():
    # Setup mock items
    items = [
        {"title": "Item A", "category": "Electronics", "rating": 4.5, "avg_sentiment": 0.8},
        {"title": "Item B", "category": "Electronics", "rating": 3.5, "avg_sentiment": 0.0},
        {"title": "Item C", "category": "Books", "rating": 2.0, "avg_sentiment": -0.5},
    ]
    item_df = pd.DataFrame(items)
    
    # Setup mock purchases
    purchases = [
        {"user_id": "user_1", "product_id": 1},
        {"user_id": "user_1", "product_id": 2},
    ]
    purchases_df = pd.DataFrame(purchases)
    item_df['id'] = [1, 2, 3]
    
    kg = KnowledgeGraphRecommender(
        item_df=item_df,
        purchases_df=purchases_df,
        n_factors=5,
        epochs=2
    )
    
    # Verify entity index mapping
    assert "Item A" in kg.entity_to_idx
    assert "Electronics" in kg.entity_to_idx
    assert "high_rating" in kg.entity_to_idx
    assert "positive_sentiment" in kg.entity_to_idx
    
    # Check if co-purchase triples are built
    co_purchased_triples = [t for t in kg.triples if t[1] == "co_purchased_with"]
    assert len(co_purchased_triples) == 2
    
    # Check if embeddings are generated
    assert kg.entity_embeddings.shape == (len(kg.entity_to_idx), 5)
    assert kg.relation_embeddings.shape == (len(kg.relation_to_idx), 5)


def test_knowledge_graph_model_training_and_similarity():
    items = [
        {"title": "Item A", "category": "Electronics", "rating": 4.5, "avg_sentiment": 0.8},
        {"title": "Item B", "category": "Electronics", "rating": 4.2, "avg_sentiment": 0.7},
        {"title": "Item C", "category": "Books", "rating": 2.0, "avg_sentiment": -0.5},
    ]
    item_df = pd.DataFrame(items)
    
    # Train TransE
    kg = KnowledgeGraphRecommender(
        item_df=item_df,
        purchases_df=None,
        n_factors=8,
        margin=1.0,
        lr=0.1,
        epochs=10
    )
    
    # Embeddings should have unit norm
    for emb in kg.entity_embeddings:
        assert np.isclose(np.linalg.norm(emb), 1.0)
        
    # Get recommendations
    recs = kg.recommend("Item A", top_n=2)
    assert len(recs) <= 2
    assert "title" in recs[0]
    assert "kg_score" in recs[0]
    # Item B should be more semantically similar to Item A (both Electronics and High Rating)
    # than Item C (Books and Low Rating/Negative Sentiment)
    item_b_sim = [r['kg_score'] for r in recs if r['title'] == 'Item B'][0]
    item_c_recs = [r['kg_score'] for r in recs if r['title'] == 'Item C']
    if item_c_recs:
        item_c_sim = item_c_recs[0]
        assert item_b_sim >= item_c_sim


def test_hybrid_recommender_with_kg():
    # Setup ContentRecommender mock
    items = [
        {"title": "Item A", "category": "Electronics", "rating": 4.5, "avg_sentiment": 0.8},
        {"title": "Item B", "category": "Electronics", "rating": 4.2, "avg_sentiment": 0.7},
        {"title": "Item C", "category": "Books", "rating": 2.0, "avg_sentiment": -0.5},
    ]
    item_df = pd.DataFrame(items)
    item_df['combined'] = item_df['title'] + " " + item_df['category']
    item_df['review_count'] = [10, 5, 2]
    
    content_model = ContentRecommender(item_df)
    
    # Train KG model
    kg = KnowledgeGraphRecommender(item_df, n_factors=5, epochs=5)
    
    # Initialize Hybrid Recommender with Content and KG models
    hybrid = HybridRecommender(
        content_model=content_model,
        collab_model=None,
        item_df=item_df,
        kg_model=kg,
        alpha=0.4,
        beta=0.0,
        gamma=0.2,
        delta=0.4
    )
    
    # Verify weights normalized
    weights = hybrid.get_weights()
    assert np.isclose(weights['alpha'] + weights['beta'] + weights['gamma'] + weights['delta'], 1.0)
    assert weights['delta'] > 0
    
    # Fetch recommendation
    recs = hybrid.recommend("Item A", explain=True, top_n=2)
    assert len(recs) > 0
    assert "kg_score" in recs[0]
    assert "hybrid_score" in recs[0]
    assert "knowledge_graph" in recs[0]["explanation"]["weighted_components"]
    assert "raw_knowledge_graph" in recs[0]["explanation"]["component_scores"]


class FakeQuery:
    def __init__(self, data):
        self.data = data

    def select(self, *args, **kwargs):
        return self

    def range(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self.data)


class FakeSupabase:
    def __init__(self, products, purchases):
        self.products = products
        self.purchases = purchases

    def table(self, name):
        if name == "products":
            return FakeQuery(self.products)
        elif name == "purchases":
            return FakeQuery(self.purchases)
        raise ValueError(f"Unknown table: {name}")


def test_api_build_with_kg(monkeypatch):
    mock_products = [
        {"id": 1, "title": "Item A", "description": "Desc A", "category": "Cat 1", "rating": 4.5, "avg_sentiment": 0.8, "review_count": 10},
        {"id": 2, "title": "Item B", "description": "Desc B", "category": "Cat 1", "rating": 4.0, "avg_sentiment": 0.6, "review_count": 5},
    ]
    mock_purchases = [
        {"user_id": "user_1", "product_id": 1, "rating": 5.0},
        {"user_id": "user_1", "product_id": 2, "rating": 4.0},
    ]
    
    fake_sb = FakeSupabase(mock_products, mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: fake_sb)
    monkeypatch.setattr(main, "get_supabase_admin", lambda: fake_sb)
    monkeypatch.setenv("ADMIN_API_TOKEN", "test-admin-token")
    
    orig_ready = main.models["ready"]
    orig_hybrid = main.models["hybrid"]
    
    try:
        from backend.csrf import CSRF_HEADER_NAME, CSRF_COOKIE_NAME, generate_csrf_token
        client = TestClient(main.app)
        csrf_token = generate_csrf_token()
        client.cookies.set(CSRF_COOKIE_NAME, csrf_token)
        response = client.post(
            "/api/build",
            headers={
                CSRF_HEADER_NAME: csrf_token,
                "x-admin-token": "test-admin-token",
            }
        )
        
        assert response.status_code == 200
        payload = response.json()
        assert "built successfully" in payload["message"]
        
        # Verify global models configured with KG Recommender
        assert main.models["ready"] is True
        assert main.models["kg"] is not None
        assert isinstance(main.models["kg"], KnowledgeGraphRecommender)
        assert main.models["hybrid"].kg_model == main.models["kg"]
    finally:
        main.models["ready"] = orig_ready
        main.models["hybrid"] = orig_hybrid
