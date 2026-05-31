"""
Unit tests for database Supabase client lazy singleton initialization.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

import src.data.db as db
from src.data.db import get_supabase, get_supabase_admin


@pytest.fixture(autouse=True)
def reset_db_singletons():
    """Reset singletons before and after each test to ensure isolation."""
    db._client = None
    db._admin_client = None
    yield
    db._client = None
    db._admin_client = None


def test_get_supabase_missing_env_raises(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"):
        get_supabase()


def test_get_supabase_admin_missing_env_fallback_none(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    admin_client = get_supabase_admin()
    assert admin_client is None


@patch("src.data.db.create_client")
def test_get_supabase_singleton_behavior(mock_create_client, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://mockurl.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "mockanonkey")

    mock_client_instance = MagicMock()
    mock_create_client.return_value = mock_client_instance

    # First call - initializes client
    client1 = get_supabase()
    assert client1 is mock_client_instance
    mock_create_client.assert_called_once_with("https://mockurl.supabase.co", "mockanonkey")

    # Second call - returns cached singleton, create_client not called again
    client2 = get_supabase()
    assert client2 is mock_client_instance
    assert mock_create_client.call_count == 1
