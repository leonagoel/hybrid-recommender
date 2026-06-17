import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api.exceptions import (
    register_exception_handlers,
    global_http_exception_handler,
    validation_exception_handler,
    global_exception_handler,
)


@pytest.fixture
def test_app():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/trigger-http")
    def trigger_http():
        raise StarletteHTTPException(status_code=404, detail="Not found")

    @app.post("/trigger-validation")
    def trigger_validation(body: dict):
        pass

    return app


def test_handlers_are_registered(test_app):
    assert StarletteHTTPException in test_app.exception_handlers
    assert RequestValidationError in test_app.exception_handlers


def test_http_exception_unified_shape(test_app):
    client = TestClient(test_app, raise_server_exceptions=False)
    response = client.get("/trigger-http")
    assert response.status_code == 404
    body = response.json()
    assert body.get("error") is True
    assert body.get("status_code") == 404
    assert isinstance(body.get("message"), str)
    assert isinstance(body.get("detail"), str)


def test_validation_error_detail_is_string(test_app):
    """'detail' must be a string — the frontend toast reads it directly."""
    client = TestClient(test_app, raise_server_exceptions=False)
    # send an integer where a dict is expected to trigger RequestValidationError
    response = client.post("/trigger-validation", json=42)
    assert response.status_code == 422
    body = response.json()
    assert body.get("error") is True
    assert body.get("status_code") == 422
    assert isinstance(body.get("detail"), str), (
        "detail must be a string so the frontend toast renders correctly"
    )
    assert isinstance(body.get("details"), list)


def test_src_api_main_calls_register_exception_handlers():
    """Verify the call-site exists in src/api/main.py without importing the full module."""
    from pathlib import Path
    source = Path("src/api/main.py").read_text()
    assert "register_exception_handlers" in source, (
        "register_exception_handlers must be imported and called in src/api/main.py"
    )
    assert "register_exception_handlers(app)" in source, (
        "register_exception_handlers(app) call is missing from src/api/main.py"
    )
