import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_get_user_recommendation_history_requires_user_or_session():
    response = client.get("/api/user/history")
    assert response.status_code == 400
    assert "Must provide user_id or session_id" in response.json()["detail"]

@patch("backend.main.get_supabase")
def test_get_user_recommendation_history_success(mock_get_supabase):
    mock_sb = MagicMock()
    mock_get_supabase.return_value = mock_sb
    
    # Mock chain: sb.table("recommendation_history").select("*").order(...).limit(...).eq(...)
    mock_query = mock_sb.table.return_value.select.return_value.order.return_value.limit.return_value.eq.return_value
    mock_query.execute.return_value.data = [{"id": "123", "user_id": "test_user"}]
    
    response = client.get("/api/user/history?user_id=test_user")
    
    assert response.status_code == 200
    assert response.json() == {"history": [{"id": "123", "user_id": "test_user"}]}
    mock_sb.table.assert_called_with("recommendation_history")
    
@patch("backend.main.get_supabase")
def test_recommendation_history_is_logged_in_paginated(mock_get_supabase):
    mock_sb = MagicMock()
    mock_get_supabase.return_value = mock_sb
    
    # Just need it to not crash.
    with patch("backend.main.MOCK_PRODUCTS", [{"title": "test item", "category": "books"}]):
        response = client.get("/api/recommend/paginated?title=test&user_id=test_user&session_id=sess1")
        assert response.status_code == 200
        
        # Verify the history logging was called
        assert mock_sb.table.called
        assert mock_sb.table.call_args[0][0] == "recommendation_history"
        assert mock_sb.table.return_value.insert.called
        insert_args = mock_sb.table.return_value.insert.call_args[0][0]
        assert insert_args["user_id"] == "test_user"
        assert insert_args["session_id"] == "sess1"
        assert insert_args["query"] == "test"
