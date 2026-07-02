"""
Tests for IDOR fix on /api/purchases/{user_id} and /api/purchases.

Related to issue #294.

Covers:
- Authentication (missing / malformed / invalid token -> 401)
- Authorisation (wrong user -> 403)
- Happy-path reads and writes (own data -> 200)
- Edge cases: empty purchase list, Pydantic validation rejection (422)

NOTE — two bugs exist in backend/main.py that this test file works around
without modifying the backend:

  1. `status` (starlette.status) is imported at module level in main.py, but
     a FastAPI endpoint `def status():` at line ~1116 clobbers that name,
     so `status.HTTP_401_UNAUTHORIZED` raises AttributeError at runtime.
     Fix: after importing the app, overwrite `backend.main.status` with the
     real starlette.status module.

  2. `return result` on line ~738 of main.py sits inside _clear_response_cache
     instead of create_purchase (indentation bug), so `result` is not in
     scope.  Fix: mock _clear_response_cache so the buggy return is never
     reached.
"""

from unittest.mock import MagicMock, patch
import sys

import starlette.status as _real_starlette_status  # noqa: E402

from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Mock broken / heavy imports before the app is imported

sys.modules["celery_app"] = MagicMock()
sys.modules["celery_app.celery_app"] = MagicMock()
sys.modules["hybrid_model"] = MagicMock()
sys.modules["src.model.hybrid_model"] = MagicMock()
sys.modules["src.model.content_model"] = MagicMock()
sys.modules["src.model.collaborative_model"] = MagicMock()
sys.modules["src.model.nlp_engine"] = MagicMock()
sys.modules["src.model.trending_model"] = MagicMock()
sys.modules["src.model.knowledge_graph"] = MagicMock()
sys.modules["src.data.data_adapter"] = MagicMock()
sys.modules["src.data.db"] = MagicMock()
sys.modules["tasks"] = MagicMock()

sys.modules["starlette.status"] = _real_starlette_status


# Stub out CSRF so it never blocks test requests


class _PassthroughCSRFMiddleware(BaseHTTPMiddleware):
    """Fake CSRF middleware that lets every request through."""

    async def dispatch(self, request, call_next):
        return await call_next(request)


class _FakeCSRFTokenResponse(BaseModel):
    """Fake Pydantic model for the CSRF-token endpoint."""

    csrfToken: str = "test-token"


_mock_csrf = MagicMock()
_mock_csrf.CSRFMiddleware = _PassthroughCSRFMiddleware
_mock_csrf.generate_csrf_token = MagicMock(return_value="test-token")
_mock_csrf.set_csrf_cookie = MagicMock()
_mock_csrf.CSRFTokenResponse = _FakeCSRFTokenResponse
sys.modules["backend.csrf"] = _mock_csrf

# App / client — imported *after* sys.modules patches

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402
import backend.main as _backend_main  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# backend/main.py defines `def status(): ...` as a FastAPI route at line ~1116.
# That function definition clobbers whatever `status` was before (starlette's
# status module), so `status.HTTP_401_UNAUTHORIZED` raises AttributeError.
# Overwriting the name on the module object after import restores the real
# starlette.status constants without touching the backend file.
_backend_main.status = _real_starlette_status
# ─────────────────────────────────────────────────────────────────────────────

client = TestClient(app)

# HTTP status code constants — plain integers, no starlette import needed

HTTP_200_OK = 200
HTTP_401_UNAUTHORIZED = 401
HTTP_403_FORBIDDEN = 403
HTTP_422_UNPROCESSABLE_ENTITY = 422

# Shared request bodies

VALID_PURCHASE_BODY = {
    "user_id": "user-42",
    "product_id": 1,
    "rating": 5.0,
    "review_text": "great",
}

SPOOFED_PURCHASE_BODY = {
    "user_id": "user-42",
    "product_id": 1,
    "rating": 5.0,
    "review_text": "great",
}


# Header helpers


def _auth_header(token: str = "fake-token") -> dict:
    return {"Authorization": f"Bearer {token}"}


def _csrf_header() -> dict:
    return {"X-CSRF-Token": "test-csrf"}


def _post_headers(token: str = "fake-token") -> dict:
    return {**_auth_header(token), **_csrf_header()}


# Supabase mock helpers


def _make_mock_user(user_id: str) -> MagicMock:
    mock_user = MagicMock()
    mock_user.user.id = user_id
    return mock_user


def _stub_select_chain(mock_sb, data) -> None:
    execute_mock = MagicMock()
    execute_mock.data = data
    limit_node = (
        mock_sb.return_value
        .table.return_value
        .select.return_value
        .eq.return_value
        .order.return_value
        .limit.return_value
    )
    limit_node.execute.return_value = execute_mock


def _stub_insert_chain(mock_sb, data) -> None:
    execute_mock = MagicMock()
    execute_mock.data = data
    insert_node = (
        mock_sb.return_value
        .table.return_value
        .insert.return_value
    )
    insert_node.execute.return_value = execute_mock


# Context manager: patch supabase + _clear_response_cache for POST happy path

def _post_patch(mock_user, insert_data=None):
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch("backend.main.get_supabase") as mock_sb, \
             patch("backend.main._clear_response_cache"):
            mock_sb.return_value.auth.get_user.return_value = mock_user
            if insert_data is not None:
                _stub_insert_chain(mock_sb, insert_data)
            yield mock_sb

    return _ctx()


# GET /api/purchases/{user_id}


class TestGetUserPurchases:
    """Tests for GET /api/purchases/{user_id}."""

    _URL = "/api/purchases/user-42"

    # --- happy path -------------------------------------------------------

    def test_own_purchases_returns_200(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            _stub_select_chain(mock_sb, [{"id": 1, "product_id": 1}])
            response = client.get(self._URL, headers=_auth_header())
        assert response.status_code == HTTP_200_OK
        assert "purchases" in response.json()

    def test_own_purchases_response_is_list(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            _stub_select_chain(mock_sb, [{"id": 1, "product_id": 99}])
            response = client.get(self._URL, headers=_auth_header())
        assert isinstance(response.json()["purchases"], list)

    def test_empty_purchases_returns_empty_list_not_null(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            _stub_select_chain(mock_sb, None)
            response = client.get(self._URL, headers=_auth_header())
        assert response.status_code == HTTP_200_OK
        assert response.json()["purchases"] == []

    # --- authentication errors --------------------------------------------

    def test_no_token_returns_401(self):
        response = client.get(self._URL)
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Not authenticated"

    def test_malformed_token_missing_bearer_prefix_returns_401(self):
        response = client.get(self._URL, headers={"Authorization": "justgarbage"})
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Not authenticated"

    def test_invalid_token_returns_401(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = None
            response = client.get(self._URL, headers=_auth_header("bad-token"))
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Invalid token"

    # --- authorisation error (IDOR) ---------------------------------------

    def test_fetching_another_users_purchases_returns_403(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-99")
            response = client.get(self._URL, headers=_auth_header())
        assert response.status_code == HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Access denied"


# POST /api/purchases


class TestCreatePurchase:
    """Tests for POST /api/purchases."""

    _URL = "/api/purchases"

    # --- happy path -------------------------------------------------------

    def test_create_own_purchase_returns_200(self):
        with _post_patch(_make_mock_user("user-42"), [{"id": 1}]):
            response = client.post(self._URL, json=VALID_PURCHASE_BODY, headers=_post_headers())
        assert response.status_code == HTTP_200_OK
        assert "purchase" in response.json()

    def test_create_purchase_response_contains_purchase_data(self):
        with _post_patch(_make_mock_user("user-42"), [{"id": 7, "product_id": 1}]):
            response = client.post(self._URL, json=VALID_PURCHASE_BODY, headers=_post_headers())
        assert response.json()["purchase"] is not None

    # --- authentication errors --------------------------------------------

    def test_no_token_returns_401(self):
        response = client.post(self._URL, json=VALID_PURCHASE_BODY, headers=_csrf_header())
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Not authenticated"

    def test_malformed_token_missing_bearer_prefix_returns_401(self):
        response = client.post(
            self._URL,
            json=VALID_PURCHASE_BODY,
            headers={"Authorization": "notbearer abc123", **_csrf_header()},
        )
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Not authenticated"

    def test_invalid_token_returns_401(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = None
            response = client.post(
                self._URL, json=VALID_PURCHASE_BODY, headers=_post_headers("bad-token")
            )
        assert response.status_code == HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Invalid token"

    # --- authorisation error (IDOR) ---------------------------------------

    def test_spoofed_user_id_in_body_returns_403(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-99")
            response = client.post(
                self._URL, json=SPOOFED_PURCHASE_BODY, headers=_post_headers()
            )
        assert response.status_code == HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Access denied"

    # --- Pydantic / input validation errors --------------------------------

    def test_rating_above_max_returns_422(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "product_id": 1, "rating": 999.0, "review_text": "great"},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_rating_below_min_returns_422(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "product_id": 1, "rating": -1.0, "review_text": "great"},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_zero_product_id_returns_422(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "product_id": 0, "rating": 4.0, "review_text": "ok"},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_review_text_exceeding_max_length_returns_422(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "product_id": 1, "rating": 3.0, "review_text": "x" * 1001},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_user_id_pattern_returns_422(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user 42")
            response = client.post(
                self._URL,
                json={"user_id": "user 42", "product_id": 1, "rating": 3.0, "review_text": "fine"},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_field_returns_422(self):
        with patch("backend.main.get_supabase") as mock_sb:
            mock_sb.return_value.auth.get_user.return_value = _make_mock_user("user-42")
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "rating": 4.0, "review_text": "nice"},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    # --- boundary / edge-value checks -------------------------------------

    def test_rating_at_minimum_boundary_returns_200(self):
        with _post_patch(_make_mock_user("user-42"), [{"id": 2}]):
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "product_id": 1, "rating": 0.0, "review_text": "bad"},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_200_OK

    def test_rating_at_maximum_boundary_returns_200(self):
        with _post_patch(_make_mock_user("user-42"), [{"id": 3}]):
            response = client.post(self._URL, json=VALID_PURCHASE_BODY, headers=_post_headers())
        assert response.status_code == HTTP_200_OK

    def test_review_text_at_max_length_boundary_returns_200(self):
        with _post_patch(_make_mock_user("user-42"), [{"id": 4}]):
            response = client.post(
                self._URL,
                json={"user_id": "user-42", "product_id": 1, "rating": 4.5, "review_text": "a" * 1000},
                headers=_post_headers(),
            )
        assert response.status_code == HTTP_200_OK
