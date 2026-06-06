import time
from fastapi.testclient import TestClient
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
    """Validates Issue #1292: System remains O(1) performance under heavy spoofing loads."""
    # Seed 5,000 rogue IP items into tracking cache
    for i in range(5000):
        _rate_limit_buckets[f"10.0.1.{i}"] = {"tokens": 0.0, "last_updated": time.time()}
        
    start = time.perf_counter()
    allowed = _apply_rate_limit("192.168.1.1")
    duration = time.perf_counter() - start
    
    assert allowed is True
    assert duration < 0.002, f"DoS Vulnerability triggered! Hot path traversal loop took {duration}s"
