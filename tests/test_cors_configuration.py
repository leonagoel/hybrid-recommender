import pytest

from backend.main import (
    ALLOWED_CORS_HEADERS,
    ALLOWED_CORS_METHODS,
    DEFAULT_CORS_ORIGINS,
    _parse_cors_origins,
)


def test_cors_origin_parser_uses_safe_defaults_when_unset():
    assert _parse_cors_origins("") == list(DEFAULT_CORS_ORIGINS)


def test_cors_origin_parser_normalizes_and_deduplicates_origins():
    assert _parse_cors_origins(
        " http://example.com/ , https://api.example.com , http://example.com "
    ) == ["http://example.com", "https://api.example.com"]


@pytest.mark.parametrize(
    "raw_value, expected_message",
    [
        ("*", "wildcard origin"),
        ("http://example.com, *", "wildcard origin"),
        ("ftp://example.com", "Invalid CORS origin"),
        ("http://example.com/path", "Invalid CORS origin"),
        ("", None),
    ],
)
def test_cors_origin_parser_rejects_unsafe_values(raw_value, expected_message):
    if expected_message is None:
        assert _parse_cors_origins(raw_value) == list(DEFAULT_CORS_ORIGINS)
        return

    with pytest.raises(RuntimeError, match=expected_message):
        _parse_cors_origins(raw_value)


def test_cors_policy_uses_explicit_allowlists():
    assert ALLOWED_CORS_METHODS == ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    assert ALLOWED_CORS_HEADERS == [
        "Accept",
        "Authorization",
        "Content-Type",
        "X-Admin-Token",
        "X-CSRF-Token",
    ]