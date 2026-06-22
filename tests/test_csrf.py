"""
tests/test_csrf.py — Tests for the Double Submit Cookie CSRF protection.

Covers:
  - Safe methods (GET, HEAD, OPTIONS) are never blocked.
  - POST / PUT / PATCH / DELETE without a cookie → 403.
  - POST / PUT / PATCH / DELETE without the header → 403.
  - POST / PUT / PATCH / DELETE with mismatched tokens → 403.
  - POST / PUT / PATCH / DELETE with matching tokens → request proceeds.
  - GET /api/csrf-token sets the cookie and returns the token in JSON.
  - generate_csrf_token() produces unique, correctly-sized tokens.
  - set_csrf_cookie() writes the expected cookie attributes.
"""

import os
import pytest
from fastapi import Response
from fastapi.testclient import TestClient

# Tell _is_secure_context() to return False so the Secure cookie flag is
# not set during tests.  TestClient uses plain HTTP; a Secure cookie would
# still be sent by TestClient (it ignores the flag), but setting this env
# var makes the behaviour explicit and matches real local-dev usage.
os.environ.setdefault("TESTING", "true")

from backend.main import app                          # noqa: E402  (after env setup)
from backend.csrf import (                            # noqa: E402
    CSRF_COOKIE_NAME,
    CSRF_HEADER_NAME,
    CSRF_TOKEN_BYTES,
    generate_csrf_token,
    set_csrf_cookie,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()                       # function scope — each test gets a clean client
def client():
    """
    Fresh TestClient per test.

    WHY function scope, not module scope?
    The previous implementation used scope="module", which shared one client
    instance across all tests.  Tests that called client.cookies.clear() would
    wipe cookies set by earlier tests in the same module, causing intermittent
    403s on tests that expected 200.  Function scope eliminates all shared state.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _inject_csrf_cookie(client: TestClient, token: str) -> None:
    """
    Inject a CSRF token directly into the TestClient cookie jar.

    This simulates what the browser does after receiving the Set-Cookie
    header from GET /api/csrf-token — it stores the cookie and sends it
    automatically on subsequent requests to the same origin.
    """
    client.cookies.set(CSRF_COOKIE_NAME, token)


# ── Token generation ──────────────────────────────────────────────────────────

def test_generate_csrf_token_returns_hex_string():
    token = generate_csrf_token()
    assert isinstance(token, str)
    # secrets.token_hex(N) produces exactly 2*N hex characters
    assert len(token) == CSRF_TOKEN_BYTES * 2


def test_generate_csrf_token_is_unique():
    # 100 tokens — the probability of a collision is astronomically small
    tokens = {generate_csrf_token() for _ in range(100)}
    assert len(tokens) == 100


# ── /api/csrf-token endpoint ──────────────────────────────────────────────────

def test_csrf_token_endpoint_returns_200(client):
    response = client.get("/api/csrf-token")
    assert response.status_code == 200


def test_csrf_token_endpoint_returns_token_in_body(client):
    response = client.get("/api/csrf-token")
    data = response.json()
    assert "csrfToken" in data
    # Must be a full 64-char hex string (256-bit token)
    assert len(data["csrfToken"]) == CSRF_TOKEN_BYTES * 2


def test_csrf_token_endpoint_sets_cookie(client):
    response = client.get("/api/csrf-token")
    # The Set-Cookie header must include our cookie name
    assert CSRF_COOKIE_NAME in response.cookies


def test_csrf_token_cookie_value_matches_body(client):
    """
    The cookie value and the JSON body value must be identical.
    This is the core invariant of the Double Submit Cookie pattern —
    the middleware compares these two values on every mutating request.
    """
    response = client.get("/api/csrf-token")
    body_token = response.json()["csrfToken"]
    cookie_token = response.cookies[CSRF_COOKIE_NAME]
    assert body_token == cookie_token


def test_csrf_token_response_has_no_cache_headers(client):
    """
    The token endpoint must set Cache-Control: no-store so proxies and
    browsers never cache a CSRF token response.  A cached response could
    allow token reuse across users or sessions.
    """
    response = client.get("/api/csrf-token")
    assert response.headers.get("cache-control") == "no-store"


# ── Safe methods are never blocked ───────────────────────────────────────────

@pytest.mark.parametrize("method,path", [
    ("GET",     "/api/health"),
    ("GET",     "/api/status"),
    ("GET",     "/api/csrf-token"),
    ("GET",     "/api/weights"),
    ("OPTIONS", "/api/health"),
])
def test_safe_methods_are_never_blocked_by_csrf(client, method, path):
    """
    GET, HEAD, and OPTIONS must never receive a CSRF 403.
    These are safe methods per RFC 7231 — they must not mutate state.
    """
    response = client.request(method, path)
    # If we get a 403, it must NOT be from our CSRF middleware
    if response.status_code == 403:
        detail = response.json().get("detail", "")
        assert "CSRF" not in detail, (
            f"{method} {path} was blocked by CSRF middleware — safe methods must never be blocked"
        )


# ── Missing cookie → 403 ─────────────────────────────────────────────────────
#
# We use /api/feedback for POST and /api/weights for PUT because:
#   - /api/feedback accepts POST and always returns 200 with valid CSRF
#   - /api/weights accepts PUT and always returns 400/200 with valid CSRF
#   - Both paths exist in the router, so FastAPI won't return 405 before
#     the middleware even runs (405 != 403, which would break the assertion)
#
# PATCH and DELETE are tested against /api/feedback and /api/weights too —
# FastAPI returns 405 for unregistered methods, but CSRFMiddleware runs
# BEFORE the router, so the middleware sees the method first and returns 403.

@pytest.mark.parametrize("method,path,body", [
    ("POST",   "/api/feedback", {"user_id": "u1", "item": "x", "feedback": "y"}),
    ("PUT",    "/api/weights",  {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}),
    ("PATCH",  "/api/feedback", {}),
    ("DELETE", "/api/feedback", {}),
])
def test_mutating_request_without_cookie_returns_403(client, method, path, body):
    """
    No CSRF cookie present → middleware must reject with 403 before the
    request reaches any route handler.
    """
    # Guarantee no cookie is present — fresh client already has none,
    # but be explicit for clarity.
    client.cookies.clear()
    response = client.request(method, path, json=body, headers={"Origin": "http://testserver"})
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


# ── Missing header → 403 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path,body", [
    ("POST",   "/api/feedback", {"user_id": "u1", "item": "x", "feedback": "y"}),
    ("PUT",    "/api/weights",  {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}),
    ("PATCH",  "/api/feedback", {}),
    ("DELETE", "/api/feedback", {}),
])
def test_mutating_request_without_header_returns_403(client, method, path, body):
    """
    Cookie present but X-CSRF-Token header absent → 403.
    The Double Submit pattern requires BOTH values.
    """
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    # Deliberately omit the X-CSRF-Token header
    response = client.request(method, path, json=body, headers={"Origin": "http://testserver"})
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


# ── Mismatched tokens → 403 ───────────────────────────────────────────────────

@pytest.mark.parametrize("method,path,body", [
    ("POST",   "/api/feedback", {"user_id": "u1", "item": "x", "feedback": "y"}),
    ("PUT",    "/api/weights",  {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}),
    ("PATCH",  "/api/feedback", {}),
    ("DELETE", "/api/feedback", {}),
])
def test_mutating_request_with_mismatched_tokens_returns_403(client, method, path, body):
    """
    Cookie token ≠ header token → 403.
    An attacker who can forge a header value but cannot read the cookie
    (cross-origin) will always produce a mismatch.
    """
    cookie_token = generate_csrf_token()
    wrong_header  = generate_csrf_token()   # different token — guaranteed by CSPRNG
    _inject_csrf_cookie(client, cookie_token)
    response = client.request(
        method, path, json=body,
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: wrong_header},
    )
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


# ── Invalid token format / length → 403 ───────────────────────────────────────

@pytest.mark.parametrize("method,path,body", [
    ("POST",   "/api/feedback", {"user_id": "u1", "item": "x", "feedback": "y"}),
    ("PUT",    "/api/weights",  {"alpha": 0.4, "beta": 0.35, "gamma": 0.25}),
    ("PATCH",  "/api/feedback", {}),
    ("DELETE", "/api/feedback", {}),
])
def test_mutating_request_with_invalid_token_length_returns_403(client, method, path, body):
    """
    Cookie token == header token, but length is invalid → 403.
    This prevents trivial token injection like "a".
    """
    short_token = "a"
    _inject_csrf_cookie(client, short_token)
    response = client.request(
        method, path, json=body,
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: short_token},
    )
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


@pytest.mark.parametrize("method,path,body", [
    ("POST",   "/api/feedback", {"user_id": "u1", "item": "x", "feedback": "y"}),
])
def test_mutating_request_with_non_hex_tokens_returns_403(client, method, path, body):
    """
    Cookie token == header token and valid length, but non-hex characters → 403.
    """
    invalid_hex_token = "Z" * (CSRF_TOKEN_BYTES * 2)
    _inject_csrf_cookie(client, invalid_hex_token)
    response = client.request(
        method, path, json=body,
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: invalid_hex_token},
    )
    assert response.status_code == 403
    assert "CSRF" in response.json()["detail"]


# ── Valid matching tokens → request proceeds past CSRF ───────────────────────

def test_post_feedback_with_valid_csrf_passes(client):
    """
    POST /api/feedback with matching cookie + header → 200.
    This is the happy path: CSRF passes and the route handler runs.
    """
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    response = client.post(
        "/api/feedback",
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: token},
        json={"user_id": "u1", "item": "Book A", "feedback": "great"},
    )
    # 200 = route handler ran; anything other than 403 means CSRF passed
    assert response.status_code == 200


def test_put_weights_with_valid_csrf_passes_csrf_check(client):
    """
    PUT /api/weights with valid CSRF → not 403.
    The route itself returns 400 (models not built), which is fine —
    it means the request reached the handler, not the CSRF guard.
    """
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    response = client.put(
        "/api/weights",
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: token},
        json={"alpha": 0.4, "beta": 0.35, "gamma": 0.25},
    )
    assert response.status_code != 403


def test_post_build_with_valid_csrf_passes_csrf_check(client):
    """
    POST /api/build with valid CSRF → not 403.
    Returns 400 (no products in DB) but CSRF is satisfied.
    """
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    response = client.post(
        "/api/build",
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: token},
        json={},
    )
    assert response.status_code != 403


def test_post_purchases_with_valid_csrf_passes_csrf_check(client):
    """
    POST /api/purchases with valid CSRF → not 403.
    May return 500 (no Supabase in test env) but CSRF is satisfied.
    """
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    response = client.post(
        "/api/purchases",
        headers={"Origin": "http://testserver", CSRF_HEADER_NAME: token},
        json={"user_id": "test-user", "product_id": 1, "rating": 4.0, "review_text": ""},
    )
    assert response.status_code != 403

def test_missing_origin_referer_returns_403(client):
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    response = client.post(
        "/api/feedback",
        headers={CSRF_HEADER_NAME: token}, # Missing Origin
        json={"user_id": "u1", "item": "Book A", "feedback": "great"},
    )
    assert response.status_code == 403
    assert "Origin and Referer missing" in response.json()["detail"]

def test_mismatched_origin_returns_403(client):
    token = generate_csrf_token()
    _inject_csrf_cookie(client, token)
    response = client.post(
        "/api/feedback",
        headers={"Origin": "http://evil-origin.com", CSRF_HEADER_NAME: token},
        json={"user_id": "u1", "item": "Book A", "feedback": "great"},
    )
    assert response.status_code == 403
    assert "cross-origin mismatch" in response.json()["detail"]


# ── set_csrf_cookie helper ────────────────────────────────────────────────────

def test_set_csrf_cookie_writes_correct_attributes():
    """
    Verify the raw Set-Cookie header produced by set_csrf_cookie contains
    the expected security attributes.
    """
    from fastapi import Response as FastAPIResponse
    resp = FastAPIResponse()
    token = generate_csrf_token()
    set_csrf_cookie(resp, token)

    raw = resp.headers.get("set-cookie", "")

    # Cookie name and value must be present
    assert CSRF_COOKIE_NAME in raw
    assert token in raw

    # SameSite must be set (lax or strict — either is acceptable)
    assert "samesite" in raw.lower()

    # HttpOnly must NOT be present.
    # The Double Submit pattern requires JS to read the cookie value.
    # HttpOnly would prevent that and break the pattern entirely.
    assert "httponly" not in raw.lower()

    # Cache-Control: no-store must be set on the response
    assert resp.headers.get("cache-control") == "no-store"
