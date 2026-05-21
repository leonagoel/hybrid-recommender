"""
Tests for API response time monitoring middleware.
"""

import logging
import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import RESPONSE_TIME_HEADER, app


def test_response_time_header_added_to_api_responses():
    client = TestClient(app)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert RESPONSE_TIME_HEADER in response.headers
    assert float(response.headers[RESPONSE_TIME_HEADER]) >= 0


def test_response_time_middleware_logs_request_metadata(caplog):
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="hybrid_recommender.api"):
        response = client.get("/api/config")

    assert response.status_code == 200
    record = next(record for record in caplog.records if record.message == "request_completed")
    assert record.method == "GET"
    assert record.path == "/api/config"
    assert record.status_code == 200
    assert record.duration_ms >= 0
