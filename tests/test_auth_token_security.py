import hmac

import pytest
from fastapi import HTTPException

from backend.auth import _require_admin_access, constant_time_token_equal


def test_constant_time_token_equal_accepts_matching_tokens():
    assert constant_time_token_equal("expected-token", "expected-token") is True


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        ("wrong-token", "expected-token"),
        ("expected-token-extra", "expected-token"),
        ("", "expected-token"),
        (None, "expected-token"),
        ("expected-token", ""),
        ("expected-token", None),
    ],
)
def test_constant_time_token_equal_rejects_invalid_tokens(provided, expected):
    assert constant_time_token_equal(provided, expected) is False


def test_constant_time_token_equal_compares_fixed_length_digests(monkeypatch):
    compared_lengths = []
    original_compare_digest = hmac.compare_digest

    def spy_compare_digest(left, right):
        compared_lengths.append((len(left), len(right)))
        return original_compare_digest(left, right)

    monkeypatch.setattr(hmac, "compare_digest", spy_compare_digest)

    assert constant_time_token_equal("short", "a-much-longer-secret-token") is False

    assert compared_lengths == [(32, 32)]


def test_require_admin_access_reads_current_env(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "first-token")
    _require_admin_access("first-token")

    monkeypatch.setenv("ADMIN_API_KEY", "rotated-token")

    with pytest.raises(HTTPException) as exc:
        _require_admin_access("first-token")

    assert exc.value.status_code == 403
    _require_admin_access("rotated-token")


def test_require_admin_access_rejects_missing_or_invalid_key(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)

    with pytest.raises(HTTPException) as missing_exc:
        _require_admin_access("anything")

    assert missing_exc.value.status_code == 500

    monkeypatch.setenv("ADMIN_API_KEY", "expected-token")

    with pytest.raises(HTTPException) as invalid_exc:
        _require_admin_access("wrong-token")

    assert invalid_exc.value.status_code == 403
