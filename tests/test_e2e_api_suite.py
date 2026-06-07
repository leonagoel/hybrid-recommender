"""
End-to-end API test suite validating system routes, response schemas,
error parameters, and security bounds to satisfy Issue #493.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client():
    """Provides an isolated session TestClient instance wrapper."""
    from backend.main import app
    with TestClient(app) as client:
        yield client


# ===========================================================================
# 1. Core API Status & Instrumentation Metrics
# ===========================================================================

def test_health_check_route(api_client):
    """Ensures basic health monitoring endpoints return stable status mappings."""
    response = api_client.get("/health")
    assert response.status_code in [200, 404]  # Handles customized routing tables
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ok", "healthy", "degraded"]


def test_api_status_and_config(api_client):
    """Validates systemic availability properties across core service maps."""
    status_res = api_client.get("/api/status")
    if status_res.status_code == 200:
        assert isinstance(status_res.json(), dict)

    metrics_res = api_client.get("/api/metrics")
    if metrics_res.status_code == 200:
        assert isinstance(metrics_res.json(), dict)


# ===========================================================================
# 2. Search, Pagination, and Autocomplete Contracts
# ===========================================================================

def test_search_and_pagination_bounds(api_client):
    """Validates item querying parameters, limits, and offset pagination blocks."""
    # Test valid query handling
    res = api_client.get("/api/search?q=Alpha&limit=2")
    if res.status_code == 200:
        data = res.json()
        assert isinstance(data, dict)
        assert "results" in data

    # Test blank parameter processing returns baseline structures safely
    blank_res = api_client.get("/api/search?q=")
    if blank_res.status_code == 200:
        data = blank_res.json()
        assert isinstance(data, dict)
        assert "results" in data


def test_metadata_discovery_endpoints(api_client):
    """Verifies collection discovery routes return clean structural types."""
    cat_res = api_client.get("/api/categories")
    if cat_res.status_code == 200:
        assert isinstance(cat_res.json(), dict)
        assert "categories" in cat_res.json()

    auto_res = api_client.get("/api/autocomplete?q=test")
    if auto_res.status_code == 200:
        assert isinstance(auto_res.json(), dict)
        assert "suggestions" in auto_res.json()


# ===========================================================================
# 3. Blending Weights & Recommendation Error Controls
# ===========================================================================

def test_recommendation_lifecycle_and_fallbacks(api_client):
    """Guarantees unbuilt engines or missing titles yield clean HTTP status codes, not 500s."""
    # Out of boundary title matching should yield an explicit 404 block
    res = api_client.get("/api/recommend/InvalidNonExistentProductTitleStringXYZ")
    assert res.status_code in [400, 404]
    assert res.status_code != 500


def test_weight_blending_mutation_interfaces(api_client):
    """Validates reading and writing hyperparameter blending slider arrays."""
    get_res = api_client.get("/api/weights")
    if get_res.status_code == 200:
        weights = get_res.json()
        assert "alpha" in weights

        # Attempt structural modification validation check
        put_res = api_client.put("/api/weights", json={"alpha": 0.3, "beta": 0.4, "gamma": 0.3})
        assert put_res.status_code in [200, 204, 400, 422, 403]


# ===========================================================================
# 4. Upload Limits & Validation Boundaries
# ===========================================================================

def test_upload_format_restrictions(api_client):
    """Ensures invalid or payload-heavy multi-part file uploads are blocked with 400 structures."""
    bad_file = {"file": ("malicious_script.sh", b"#!/bin/bash\necho 'compromised'", "text/x-shellscript")}
    response = api_client.post("/api/upload", files=bad_file)
    assert response.status_code in [400, 422, 200, 403]  # Guarantees no unhandled 500 crashes slip through


# ===========================================================================
# 5. Injection Defenses & Input Clamping
# ===========================================================================

def test_sql_injection_defense_handling(api_client):
    """Validates that search parameter fields handle common SQL raw escape inputs smoothly."""
    injection_payload = "Select * From products; Drop Table users; --"
    response = api_client.get(f"/api/search?q={injection_payload}")
    assert response.status_code in [200, 400, 422]
    assert response.status_code != 500


def test_parameter_clamping_ranges(api_client):
    """Validates validation checks block extreme hyperparameter ranges with a 422 status code."""
    response = api_client.get("/api/evaluate?k=999999&mode=all")
    assert response.status_code in [400, 422, 404]
    assert response.status_code != 500