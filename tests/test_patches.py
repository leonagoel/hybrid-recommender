import time
from types import SimpleNamespace
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from backend.main import app, _rate_limit_buckets, _apply_rate_limit

client = TestClient(app)

def test_xai_explanation_endpoint_integrity():
    """Validates Issue #1315: XAI endpoint handles exactly 100% total bounds."""
    response = client.get("/api/recommendations/product_99/explanation?user_id=user_12")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"

    pct = json_data["data"]["breakdown_percentages"]
    assert pct["content"] + pct["collaborative"] + pct["sentiment"] == 100

def test_rate_limiter_dos_mitigation_speed():
    """Validates Issue #1292 + #887: Token Bucket remains O(1) under heavy IP spoofing.

    Seeds 5,000 unique IPs with exhausted buckets (tokens=0) and verifies that
    processing a new request still completes in under 2 ms — proving the LRU eviction
    strategy keeps the hot path constant-time regardless of bucket table size.
    """
    now = time.time()
    # Seed with Token Bucket dict format (tokens, last_updated)
    for i in range(5000):
        _rate_limit_buckets[("search", f"10.0.1.{i}")] = {
            "tokens": 0.0,
            "last_updated": now,
        }

    # Build minimal fake request/response objects
    fake_request = SimpleNamespace(client=SimpleNamespace(host="192.168.1.1"))
    fake_response = SimpleNamespace(headers={})

    start = time.perf_counter()
    result = _apply_rate_limit(
        fake_request,
        fake_response,
        scope="search",
        limit_env="RATE_LIMIT_SEARCH_PER_MIN",
        default_limit=60,
    )
    duration = time.perf_counter() - start

    # New IP with full bucket — must be allowed
    assert result is None, "Expected request to be allowed for a fresh IP"
    assert duration < 0.002, f"DoS Vulnerability triggered! Hot path took {duration:.4f}s"
