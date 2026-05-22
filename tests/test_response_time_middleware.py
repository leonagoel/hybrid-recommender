import logging

from fastapi.testclient import TestClient

from backend.main import (
    app,
    get_response_metrics_snapshot,
    record_response_metric,
    reset_response_metrics,
)


client = TestClient(app)


def setup_function():
    reset_response_metrics()


def test_response_time_header_is_added_to_api_responses():
    response = client.get("/api/config")

    assert response.status_code == 200
    assert "x-response-time" in response.headers
    assert response.headers["x-response-time"].endswith("ms")


def test_metrics_endpoint_reports_request_stats_and_error_rate():
    client.get("/api/config")
    client.get("/api/does-not-exist")

    response = client.get("/api/metrics")
    payload = response.json()

    assert response.status_code == 200
    assert payload["total_requests"] >= 2
    assert payload["avg_response_time"] >= 0
    assert payload["p95_response_time"] >= 0
    assert payload["error_rate"] > 0


def test_metrics_snapshot_calculates_average_p95_and_error_rate():
    record_response_metric("/api/fast", "GET", 200, 10.0)
    record_response_metric("/api/slow", "POST", 503, 800.0)

    snapshot = get_response_metrics_snapshot()

    assert snapshot["total_requests"] == 2
    assert snapshot["avg_response_time"] == 405.0
    assert snapshot["p95_response_time"] == 800.0
    assert snapshot["error_rate"] == 50.0


def test_slow_responses_are_logged_as_warnings(caplog):
    with caplog.at_level(logging.WARNING):
        record_response_metric(
            "/api/recommend/test",
            "GET",
            200,
            501.0,
        )

    assert "response_time_ms=501.00" in caplog.text
    assert "endpoint=/api/recommend/test" in caplog.text
