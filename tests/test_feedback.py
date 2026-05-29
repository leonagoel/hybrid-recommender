import os
os.environ["TESTING"] = "true" 

from fastapi.testclient import TestClient
from backend import main

client = TestClient(main.app)

def get_csrf_token():
    """
    Helper function to get CSRF token.
    its imp to set Cookie and  header .
    """
    response = client.get("/api/csrf-token")
    token = response.json()["csrfToken"]
    client.cookies.set("csrftoken", token)  # Cookie bhi set karo
    return token

class _FakeInsertResult:
    def __init__(self, data):
        self.data = data


class _FakeFeedbackTable:
    def __init__(self):
        self.inserted = []

    def insert(self, payload):
        self.inserted.append(payload)
        return self

    def execute(self):
        return _FakeInsertResult([self.inserted[-1]])


class _FakeSupabase:
    def __init__(self):
        self.feedback_table = _FakeFeedbackTable()

    def table(self, name):
        assert name == "feedback_submissions"
        return self.feedback_table


class _FakeInsertResult:
    def __init__(self, data):
        self.data = data


class _FakeFeedbackTable:
    def __init__(self):
        self.inserted = []

    def insert(self, payload):
        self.inserted.append(payload)
        return self

    def execute(self):
        return _FakeInsertResult([self.inserted[-1]])


class _FakeSupabase:
    def __init__(self):
        self.feedback_table = _FakeFeedbackTable()

    def table(self, name):
        assert name == "feedback_submissions"
        return self.feedback_table


def test_submit_feedback_validation_failures():
    #Test: Invalid inputs should return 422 (Validation Error).Empty user_id, item, feedback — should fail.
    token = get_csrf_token()
    headers = {"x-csrf-token": token}
    print("Token:", token)  # debug
    print("Headers:", headers)  # debug
    # Empty user_id should fail
    response = client.post("/api/feedback", json={"user_id": "", "item": "item1", "feedback": "Good"})
    assert response.status_code == 422

    # Empty item should fail
    response = client.post("/api/feedback", json={"user_id": "user123", "item": "", "feedback": "Good"})
    assert response.status_code == 422

    # Empty feedback should fail
    response = client.post("/api/feedback", json={"user_id": "user123", "item": "item1", "feedback": ""})
    assert response.status_code == 422


def test_submit_feedback_success(monkeypatch):
    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(main, "get_supabase_admin", lambda: fake_supabase)

    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Excellent service!","thumbs": "up"}
    )
    print("Response:", response.json()) 
    assert response.status_code == 200
    
    payload = response.json()
    assert "message" in payload
    assert payload["message"] == "Feedback submitted successfully"
    assert payload["feedback"]["user_id"] == "user123"
    assert payload["feedback"]["item"] == "item1"
    assert payload["feedback"]["feedback"] == "Excellent service!"
    assert "created_at" in payload["feedback"]
    assert payload["feedback"]["metadata"]["source_ip"] is not None
    assert fake_supabase.feedback_table.inserted[0]["user_id"] == "user123"
    assert fake_supabase.feedback_table.inserted[0]["item"] == "item1"
    assert fake_supabase.feedback_table.inserted[0]["feedback"] == "Excellent service!"


def test_submit_feedback_fails_when_storage_unavailable(monkeypatch):
    monkeypatch.setattr(main, "get_supabase_admin", lambda: None)

    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Excellent service!"}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Feedback storage is unavailable."
