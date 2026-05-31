import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.federated_learning import FederatedClient, FederatedServer


class TestFederatedClient:
    def test_init(self):
        client = FederatedClient("user1", {"item1": 4.0, "item2": 3.5})
        assert client.user_id == "user1"
        assert client.private_ratings == {"item1": 4.0, "item2": 3.5}
        assert client.user_factor is None

    def test_compute_local_user_factor_no_ratings(self):
        client = FederatedClient("user1", {})
        global_item_factors = np.random.randn(10, 5)
        title_to_idx = {"item1": 0, "item2": 1}
        result = client.compute_local_user_factor(global_item_factors, title_to_idx, n_factors=10)
        assert np.allclose(result, np.zeros(10))

    def test_compute_local_user_factor_unrated_items(self):
        client = FederatedClient("user1", {"item3": 4.0})
        global_item_factors = np.random.randn(10, 5)
        title_to_idx = {"item1": 0, "item2": 1}
        result = client.compute_local_user_factor(global_item_factors, title_to_idx, n_factors=10)
        assert np.allclose(result, np.zeros(10))

    def test_compute_local_user_factor_with_ratings(self):
        client = FederatedClient("user1", {"item1": 4.0, "item2": 3.0})
        global_item_factors = np.eye(10, 2)
        title_to_idx = {"item1": 0, "item2": 1}
        result = client.compute_local_user_factor(global_item_factors, title_to_idx, n_factors=10)
        assert result.shape == (10,)
        assert client.user_factor is not None

    def test_compute_local_item_updates_no_user_factor(self):
        client = FederatedClient("user1", {"item1": 4.0})
        global_item_factors = np.eye(10, 2)
        title_to_idx = {"item1": 0, "item2": 1}
        updates = client.compute_local_item_updates(global_item_factors, title_to_idx)
        assert updates == {}

    def test_compute_local_item_updates_with_user_factor(self):
        client = FederatedClient("user1", {"item1": 4.0})
        global_item_factors = np.eye(10, 2)
        title_to_idx = {"item1": 0, "item2": 1}
        client.user_factor = np.ones(10)
        updates = client.compute_local_item_updates(global_item_factors, title_to_idx)
        assert "item1" in updates
        assert updates["item1"].shape == (10,)


class TestFederatedServer:
    def test_init(self):
        server = FederatedServer(item_list=["item1", "item2", "item3"], n_factors=10)
        assert server.n_factors == 10
        assert len(server.item_list) == 3
        assert server.global_item_factors is not None

    def test_initialize_global_factors(self):
        server = FederatedServer(item_list=["item1", "item2", "item3"], n_factors=10)
        item_factors = np.random.randn(10, 3)
        server.global_item_factors = item_factors
        assert server.global_item_factors is not None
        assert server.global_item_factors.shape == (10, 3)

    def test_aggregate_updates_empty(self):
        server = FederatedServer(item_list=["item1", "item2"], n_factors=10)
        server.global_item_factors = np.random.randn(10, 2)
        server.aggregate_updates([])
        assert len(server.item_list) == 2

    def test_apply_updates_no_updates(self):
        server = FederatedServer(item_list=["item1", "item2"], n_factors=10)
        original = np.random.randn(10, 2)
        server.global_item_factors = original.copy()
        server.aggregate_updates([])
        assert np.allclose(server.global_item_factors, original)
