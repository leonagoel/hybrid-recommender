"""
Tests for A/B Registry, Canary Routing, and Shadow model metrics.
"""
import os
import sys
import time
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from fastapi.testclient import TestClient

# Set up fake env variables before importing main to prevent RuntimeError
os.environ["SUPABASE_URL"] = "https://fake-supabase.co"
os.environ["SUPABASE_ANON_KEY"] = "fake-anon-key"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import main as backend_main
from backend.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME


class _FakeQuery:
    def __init__(self, table_name, dataset):
        self._table = table_name
        self._dataset = dataset
        self._filters = []

    def select(self, columns, count=None):
        return self

    def range(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def execute(self):
        rows = list(self._dataset.get(self._table, []))
        return SimpleNamespace(data=rows, count=len(rows))


class _FakeSupabase:
    def __init__(self, dataset):
        self._dataset = dataset

    def table(self, name):
        return _FakeQuery(name, self._dataset)


@pytest.fixture
def client():
    with TestClient(backend_main.app, raise_server_exceptions=True) as c:
        yield c

@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    # Save original values
    orig_registry = dict(backend_main.MODEL_REGISTRY)
    orig_active = backend_main.ACTIVE_MODEL_VERSION
    orig_shadow = backend_main.SHADOW_MODEL_VERSION
    orig_staging = backend_main.STAGING_MODEL_VERSION
    orig_traffic = backend_main.CANARY_TRAFFIC_PERCENT
    orig_threshold = backend_main.CANARY_ERROR_THRESHOLD
    orig_logs = list(backend_main.SHADOW_LOGS)

    # Set up mock dataset
    dataset = {
        'products': [
            {'id': 1, 'title': 'Acoustic Noise-Cancelling Headphones', 'category': 'Electronics', 'rating': 4.8, 'avg_sentiment': 0.8, 'review_count': 100, 'description': 'Noise cancelling headphones'},
            {'id': 2, 'title': 'Ergonomic Mechanical Keyboard', 'category': 'Electronics', 'rating': 4.5, 'avg_sentiment': 0.7, 'review_count': 80, 'description': 'Mechanical keyboard'},
        ],
        'purchases': [
            {'id': 10, 'user_id': 'u1', 'product_id': 1},
        ],
    }
    fake = _FakeSupabase(dataset)
    monkeypatch.setattr(backend_main, 'get_supabase', lambda: fake)
    monkeypatch.setenv(backend_main.ADMIN_API_TOKEN_ENV, "test-admin-token")

    yield

    # Restore original values
    backend_main.MODEL_REGISTRY = orig_registry
    backend_main.ACTIVE_MODEL_VERSION = orig_active
    backend_main.SHADOW_MODEL_VERSION = orig_shadow
    backend_main.STAGING_MODEL_VERSION = orig_staging
    backend_main.CANARY_TRAFFIC_PERCENT = orig_traffic
    backend_main.CANARY_ERROR_THRESHOLD = orig_threshold
    backend_main.SHADOW_LOGS = orig_logs

def _get_csrf_headers(client, token="test-admin-token"):
    res = client.get("/api/csrf-token")
    csrf_token = res.json()["csrfToken"]
    return {
        "Authorization": f"Bearer {token}",
        CSRF_HEADER_NAME: csrf_token,
        "Origin": "http://testserver",
    }

def test_canary_settings_endpoints(client):
    headers = _get_csrf_headers(client)
    
    # Test GET
    get_res = client.get("/api/admin/models/canary_settings")
    assert get_res.status_code == 200
    assert "traffic_percent" in get_res.json()
    assert "error_threshold" in get_res.json()
    
    # Test POST
    post_res = client.post(
        "/api/admin/models/canary_settings",
        json={"traffic_percent": 25, "error_threshold": 0.10},
        headers=headers,
    )
    assert post_res.status_code == 200
    assert backend_main.CANARY_TRAFFIC_PERCENT == 25
    assert backend_main.CANARY_ERROR_THRESHOLD == 0.10

def test_register_promote_shadow_and_delete_flow(client):
    headers = _get_csrf_headers(client)
    
    # Ensure models have data populated or use mockup adapt logic
    backend_main.models["ready"] = True
    
    # 1. Register a new model in staging
    reg_res = client.post(
        "/api/admin/models/register",
        json={"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "version": "test-v1"},
        headers=headers,
    )
    assert reg_res.status_code == 200
    assert reg_res.json()["version"] == "test-v1"
    assert backend_main.STAGING_MODEL_VERSION == "test-v1"
    
    # 2. Promote model to production
    promote_res = client.post(
        "/api/admin/models/test-v1/promote",
        headers=headers,
    )
    assert promote_res.status_code == 200
    assert backend_main.ACTIVE_MODEL_VERSION == "test-v1"
    assert backend_main.STAGING_MODEL_VERSION is None
    
    # 3. Register a second model in staging
    reg_res2 = client.post(
        "/api/admin/models/register",
        json={"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "version": "test-v2"},
        headers=headers,
    )
    assert reg_res2.status_code == 200
    assert backend_main.STAGING_MODEL_VERSION == "test-v2"
    
    # 4. Set staging model as shadow
    shadow_res = client.post(
        "/api/admin/models/test-v2/shadow",
        headers=headers,
    )
    assert shadow_res.status_code == 200
    assert backend_main.SHADOW_MODEL_VERSION == "test-v2"
    assert backend_main.STAGING_MODEL_VERSION is None
    
    # 5. List models
    list_res = client.get("/api/admin/models")
    assert list_res.status_code == 200
    versions = [m["version"] for m in list_res.json()["models"]]
    assert "test-v1" in versions
    assert "test-v2" in versions
    
    # 6. Delete test-v2
    del_res = client.delete("/api/admin/models/test-v2", headers=headers)
    assert del_res.status_code == 200
    assert "test-v2" not in backend_main.MODEL_REGISTRY
    assert backend_main.SHADOW_MODEL_VERSION is None

def test_canary_traffic_routing_and_auto_rollback(client):
    headers = _get_csrf_headers(client)
    
    # Register production and staging models
    client.post(
        "/api/admin/models/register",
        json={"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "version": "prod-v"},
        headers=headers,
    )
    client.post(
        "/api/admin/models/prod-v/promote",
        headers=headers,
    )
    
    client.post(
        "/api/admin/models/register",
        json={"alpha": 0.6, "beta": 0.3, "gamma": 0.1, "version": "stage-v"},
        headers=headers,
    )
    
    # Set canary traffic to 100% to ensure all traffic goes to canary
    client.post(
        "/api/admin/models/canary_settings",
        json={"traffic_percent": 100, "error_threshold": 0.05},
        headers=headers,
    )
    
    # Route recommendation query and check it goes to stage-v
    backend_main.models["ready"] = True
    
    # Query recommend API (canary)
    rec_res = client.get("/api/recommend?title=Acoustic Noise-Cancelling Headphones")
    assert rec_res.status_code == 200
    
    # Let's verify that metrics incremented in stage-v
    metrics = backend_main.MODEL_REGISTRY["stage-v"]["metrics"]
    assert metrics["canary_requests"] > 0
    
    # Now simulate errors to trigger auto-rollback.
    original_recommend = backend_main.MODEL_REGISTRY["stage-v"]["hybrid"].recommend
    def failing_recommend(*args, **kwargs):
        raise ValueError("Simulated canary crash")
    
    backend_main.MODEL_REGISTRY["stage-v"]["hybrid"].recommend = failing_recommend
    
    # Make 5 requests. They should all raise an exception (which translates to 500 error responses in route)
    for _ in range(5):
        backend_main._clear_response_cache()
        try:
            client.get("/api/recommend?title=Acoustic Noise-Cancelling Headphones")
        except ValueError:
            pass
            
    # Verify auto-rollback occurred
    assert backend_main.STAGING_MODEL_VERSION is None
    assert backend_main.MODEL_REGISTRY["stage-v"]["status"] == "rolled_back"

def test_shadow_mode_metrics_calculation(client):
    headers = _get_csrf_headers(client)
    
    # Set up production (active) and shadow models
    client.post(
        "/api/admin/models/register",
        json={"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "version": "active-v"},
        headers=headers,
    )
    client.post(
        "/api/admin/models/active-v/promote",
        headers=headers,
    )
    
    client.post(
        "/api/admin/models/register",
        json={"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "version": "shadow-v"},
        headers=headers,
    )
    client.post(
        "/api/admin/models/shadow-v/shadow",
        headers=headers,
    )
    
    backend_main.models["ready"] = True
    backend_main.SHADOW_LOGS.clear()
    
    # Request recommendations which should run shadow evaluation
    rec_res = client.get("/api/recommend?title=Acoustic Noise-Cancelling Headphones")
    assert rec_res.status_code == 200
    
    # Fetch comparison logs
    comp_res = client.get("/api/admin/models/shadow_comparison", headers=headers)
    assert comp_res.status_code == 200
    data = comp_res.json()
    assert data["total_comparisons"] == 1
    assert "avg_agreement_rate" in data
    assert "avg_ndcg_delta" in data
    assert len(data["logs"]) == 1
    assert data["logs"][0]["shadow_version"] == "shadow-v"
