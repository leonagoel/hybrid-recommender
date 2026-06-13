from fastapi.testclient import TestClient
from types import SimpleNamespace
from backend import main

client = TestClient(main.app)

class FakeQuery:
    def __init__(self, data):
        self.data = data

    def update(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self.data)

class FakeSupabase:
    def __init__(self, table_query):
        self.table_query = table_query

    def table(self, name):
        assert name == "purchases"
        return self.table_query

def test_register_and_merge_history_success(monkeypatch):
    mock_response = [{"id": 1, "user_id": "registered_user_123", "product_id": 123}]
    query_mock = FakeQuery(mock_response)
    monkeypatch.setattr(main, "get_supabase", lambda: FakeSupabase(query_mock))

    # Seed an in-memory interaction for the guest to test memory updates
    main.USER_INTERACTIONS.append({
        "user_id": "guest_token_abc",
        "item_id": 123,
        "interaction_type": "view",
        "timestamp": "2026-06-13T12:00:00Z"
    })

    # Prepare CSRF cookies and headers for the request
    token = "a" * 64
    client.cookies.set("csrftoken", token)
    response = client.post(
        "/api/register",
        json={"guest_id": "guest_token_abc", "user_id": "registered_user_123"},
        headers={"X-CSRF-Token": token}
    )
    assert response.status_code == 200
    
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["updated_count"] == 1
    
    # Verify in-memory USER_INTERACTIONS got updated
    assert main.USER_INTERACTIONS[-1]["user_id"] == "registered_user_123"
