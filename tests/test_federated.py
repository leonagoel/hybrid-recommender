"""
Unit and integration tests for federated learning modules and APIs.
"""
import pytest
import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.model.federated_learning import FederatedClient, FederatedServer, train_federated_collaborative_model
from backend import main


def test_federated_client_factor_calculation():
    # Setup global item factors and lookup
    n_factors = 3
    title_to_idx = {"Item A": 0, "Item B": 1}
    # 3 latent components, 2 items
    global_item_factors = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])
    
    # Client rated Item A 4.0, but hasn't rated Item B
    client = FederatedClient(user_id="user1", private_ratings={"Item A": 4.0})
    
    # Calculate local user factor
    user_factor = client.compute_local_user_factor(
        global_item_factors=global_item_factors,
        title_to_idx=title_to_idx,
        n_factors=n_factors,
        reg=0.05
    )
    
    # Must be 1D array of shape [n_factors]
    assert user_factor.shape == (n_factors,)
    
    # If the user has no rated items in the global vocabulary
    client_empty = FederatedClient(user_id="user2", private_ratings={"Item Unknown": 5.0})
    empty_factor = client_empty.compute_local_user_factor(
        global_item_factors=global_item_factors,
        title_to_idx=title_to_idx,
        n_factors=n_factors,
        reg=0.05
    )
    assert np.allclose(empty_factor, 0.0)


def test_federated_client_item_updates():
    n_factors = 3
    title_to_idx = {"Item A": 0, "Item B": 1}
    global_item_factors = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])
    
    client = FederatedClient(user_id="user1", private_ratings={"Item A": 4.0})
    # Compute factor first
    client.compute_local_user_factor(global_item_factors, title_to_idx, n_factors, reg=0.05)
    
    updates = client.compute_local_item_updates(global_item_factors, title_to_idx, reg=0.05)
    
    # Updates should contain 'Item A' but NOT 'Item B' because the user only rated 'Item A'
    assert "Item A" in updates
    assert "Item B" not in updates
    assert updates["Item A"].shape == (n_factors,)


def test_federated_server_aggregation():
    item_list = ["Item A", "Item B"]
    n_factors = 3
    
    server = FederatedServer(item_list=item_list, n_factors=n_factors, learning_rate=0.1, reg=0.05)
    
    # Store initial global item factors
    initial_factors = server.global_item_factors.copy()
    
    # Simulate client updates:
    # Client 1 updates Item A
    # Client 2 updates Item A and Item B
    client1_updates = {"Item A": np.array([0.1, 0.2, 0.3])}
    client2_updates = {"Item A": np.array([0.3, 0.4, 0.5]), "Item B": np.array([0.2, 0.4, 0.6])}
    
    server.aggregate_updates([client1_updates, client2_updates])
    
    # Updated global item factors
    # For Item A (idx 0), average update = mean([0.1,0.2,0.3] and [0.3,0.4,0.5]) = [0.2, 0.3, 0.4]
    # New Item A factor = initial_A + lr * (average_update - reg * initial_A)
    expected_A_update = np.array([0.2, 0.3, 0.4])
    assert np.allclose(
        server.global_item_factors[:, 0],
        initial_factors[:, 0] + 0.1 * (expected_A_update - 0.05 * initial_factors[:, 0])
    )
    
    # For Item B (idx 1), average update = [0.2, 0.4, 0.6]
    expected_B_update = np.array([0.2, 0.4, 0.6])
    assert np.allclose(
        server.global_item_factors[:, 1],
        initial_factors[:, 1] + 0.1 * (expected_B_update - 0.05 * initial_factors[:, 1])
    )


def test_train_federated_collaborative_model():
    # Make a small dummy interaction dataframe
    data = {
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "title": ["Item A", "Item B", "Item B", "Item C", "Item A", "Item C"],
        "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 2.0]
    }
    df = pd.DataFrame(data)
    
    recommender = train_federated_collaborative_model(
        interaction_df=df,
        n_factors=2,
        epochs=3,
        lr=0.1,
        reg=0.05
    )
    
    # Verify the recommender structure
    assert recommender.item_factors.shape == (2, 3) # 2 factors, 3 unique items
    assert recommender.user_factors.shape == (3, 2) # 3 unique users, 2 factors
    assert len(recommender.title_list) == 3
    
    # Test recommendations can be retrieved
    recs = recommender.recommend("Item A", top_n=2)
    assert len(recs) > 0
    assert "title" in recs[0]
    assert "collab_score" in recs[0]


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


def test_api_train_federated_endpoint_success(monkeypatch):
    # Setup mock DB data
    mock_products = [
        {"id": 1, "title": "Item A", "description": "Desc A", "category": "Cat 1", "rating": 4.5, "avg_sentiment": 0.8, "review_count": 10},
        {"id": 2, "title": "Item B", "description": "Desc B", "category": "Cat 1", "rating": 4.0, "avg_sentiment": 0.6, "review_count": 5},
        {"id": 3, "title": "Item C", "description": "Desc C", "category": "Cat 2", "rating": 5.0, "avg_sentiment": 0.9, "review_count": 15},
    ]
    # At least 11 interactions, at least 2 unique users
    mock_purchases = [
        {"user_id": "user_1", "product_id": 1, "rating": 5.0},
        {"user_id": "user_1", "product_id": 2, "rating": 4.0},
        {"user_id": "user_1", "product_id": 3, "rating": 3.0},
        {"user_id": "user_2", "product_id": 1, "rating": 4.0},
        {"user_id": "user_2", "product_id": 2, "rating": 5.0},
        {"user_id": "user_2", "product_id": 3, "rating": 4.0},
        {"user_id": "user_3", "product_id": 1, "rating": 3.0},
        {"user_id": "user_3", "product_id": 2, "rating": 2.0},
        {"user_id": "user_3", "product_id": 3, "rating": 5.0},
        {"user_id": "user_4", "product_id": 1, "rating": 4.0},
        {"user_id": "user_4", "product_id": 3, "rating": 4.0},
    ]
    
    fake_sb = FakeSupabase(mock_products, mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: fake_sb)
    
    # Save the original state of models to restore after test
    orig_ready = main.models["ready"]
    orig_hybrid = main.models["hybrid"]
    
    try:
        client = TestClient(main.app)
        response = client.post(
            "/api/train/federated",
            json={"n_factors": 2, "epochs": 2, "lr": 0.1, "reg": 0.01}
        )
        
        assert response.status_code == 200
        payload = response.json()
        assert "trained successfully" in payload["message"]
        assert payload["items"] == 3
        assert payload["users"] == 4
        assert "build_time_seconds" in payload
        
        # Verify global model state updated
        assert main.models["ready"] is True
        assert main.models["collab"] is not None
        assert main.models["hybrid"] is not None
    finally:
        # Restore state
        main.models["ready"] = orig_ready
        main.models["hybrid"] = orig_hybrid


def test_api_train_federated_endpoint_not_enough_data(monkeypatch):
    mock_products = [
        {"id": 1, "title": "Item A", "description": "Desc A", "category": "Cat 1", "rating": 4.5, "avg_sentiment": 0.8, "review_count": 10},
    ]
    # Only 2 purchases (needs at least 11)
    mock_purchases = [
        {"user_id": "user_1", "product_id": 1, "rating": 5.0},
        {"user_id": "user_2", "product_id": 1, "rating": 4.0},
    ]
    
    fake_sb = FakeSupabase(mock_products, mock_purchases)
    monkeypatch.setattr(main, "get_supabase", lambda: fake_sb)
    
    client = TestClient(main.app)
    response = client.post(
        "/api/train/federated",
        json={"n_factors": 2, "epochs": 2, "lr": 0.1, "reg": 0.01}
    )
    
    assert response.status_code == 400
    assert "Not enough interaction data" in response.json()["detail"]
