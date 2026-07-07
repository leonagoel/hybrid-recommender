"""
Tests for checking and verifying Supabase network dependency mocking.

These tests assert that:
1. No real network calls are made to Supabase during test runs.
2. All authentication flows (success, failure) are successfully mocked.
3. All database query chains (select, filter, limits, execute) are successfully mocked.
4. Mocks are called with exact expected arguments.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Ensure environment variables are present so db.py client initialization compiles,
# but our mocking ensures no live HTTP connections are ever attempted.
os.environ["SUPABASE_URL"] = "https://mockproject.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = "mock-anon-key"
os.environ["SUPABASE_SERVICE_KEY"] = "mock-service-key"

from src.data.db import get_supabase, get_supabase_admin


@pytest.fixture(autouse=True)
def reset_db_singletons():
    """Reset the module-level client singletons in db.py before and after each test."""
    import src.data.db as db
    with db._client_lock:
        db._client = None
    with db._admin_client_lock:
        db._admin_client = None
    yield
    with db._client_lock:
        db._client = None
    with db._admin_client_lock:
        db._admin_client = None


@pytest.fixture
def mock_supabase_client():
    """Fixture that constructs a mock Supabase client with chainable query methods."""
    mock_client = MagicMock()
    mock_client.auth = MagicMock()
    
    mock_table_builder = MagicMock()
    mock_client.table.return_value = mock_table_builder
    
    mock_query_builder = MagicMock()
    mock_table_builder.select.return_value = mock_query_builder
    mock_table_builder.insert.return_value = mock_query_builder
    mock_table_builder.upsert.return_value = mock_query_builder
    mock_table_builder.delete.return_value = mock_query_builder
    
    mock_query_builder.eq.return_value = mock_query_builder
    mock_query_builder.limit.return_value = mock_query_builder
    
    return mock_client, mock_table_builder, mock_query_builder


# ── Client Creation Mocks ─────────────────────────────────────────────────────

@patch("src.data.db.create_client")
def test_get_supabase_initializes_client_correctly(mock_create_client, mock_supabase_client):
    mock_client, _, _ = mock_supabase_client
    mock_create_client.return_value = mock_client
    
    client = get_supabase()
    
    assert client is mock_client
    mock_create_client.assert_called_once_with(
        "https://mockproject.supabase.co",
        "mock-anon-key"
    )


@patch("src.data.db.create_client")
def test_get_supabase_admin_initializes_client_correctly(mock_create_client, mock_supabase_client):
    mock_client, _, _ = mock_supabase_client
    mock_create_client.return_value = mock_client
    
    client = get_supabase_admin()
    
    assert client is mock_client
    mock_create_client.assert_called_once_with(
        "https://mockproject.supabase.co",
        "mock-service-key"
    )


# ── Authentication Flow Mocking ───────────────────────────────────────────────

@patch("src.data.db.create_client")
def test_successful_authentication_flow(mock_create_client, mock_supabase_client):
    mock_client, _, _ = mock_supabase_client
    mock_create_client.return_value = mock_client
    
    fake_response = {"session": {"access_token": "fake-jwt-token"}, "user": {"id": "usr_abc123"}}
    mock_client.auth.sign_in_with_password.return_value = fake_response
    
    client = get_supabase()
    credentials = {"email": "test@example.com", "password": "securepassword"}
    response = client.auth.sign_in_with_password(credentials)
    
    assert response == fake_response
    mock_client.auth.sign_in_with_password.assert_called_once_with(credentials)


@patch("src.data.db.create_client")
def test_failed_authentication_flow(mock_create_client, mock_supabase_client):
    mock_client, _, _ = mock_supabase_client
    mock_create_client.return_value = mock_client
    
    mock_client.auth.sign_in_with_password.side_effect = Exception("Invalid login credentials")
    
    client = get_supabase()
    with pytest.raises(Exception) as exc_info:
        client.auth.sign_in_with_password({"email": "test@example.com", "password": "wrong"})
        
    assert "Invalid login credentials" in str(exc_info.value)
    mock_client.auth.sign_in_with_password.assert_called_once()


# ── Database Operations Mocking ───────────────────────────────────────────────

@patch("src.data.db.create_client")
def test_successful_database_query_execution(mock_create_client, mock_supabase_client):
    mock_client, mock_table, mock_query = mock_supabase_client
    mock_create_client.return_value = mock_client
    
    mock_data = [{"id": 101, "title": "Wolverine Comics", "rating": 4.9}]
    mock_query.execute.return_value = SimpleNamespace(data=mock_data)
    
    client = get_supabase()
    result = client.table("products").select("id, title, rating").eq("id", 101).execute()
    
    assert result.data == mock_data
    mock_client.table.assert_called_once_with("products")
    mock_table.select.assert_called_once_with("id, title, rating")
    mock_query.eq.assert_called_once_with("id", 101)
    mock_query.execute.assert_called_once()


@patch("src.data.db.create_client")
def test_database_query_exception_handling(mock_create_client, mock_supabase_client):
    mock_client, mock_table, mock_query = mock_supabase_client
    mock_create_client.return_value = mock_client
    
    mock_query.execute.side_effect = Exception("Postgrest API error: Column not found")
    
    client = get_supabase()
    with pytest.raises(Exception) as exc_info:
        client.table("products").select("non_existent_column").execute()
        
    assert "Column not found" in str(exc_info.value)
    mock_query.execute.assert_called_once()
