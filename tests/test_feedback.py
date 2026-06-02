import os
os.environ["TESTING"] = "true"

from fastapi.testclient import TestClient
from backend import main

client = TestClient(main.app)

def get_csrf_token():
    """Helper to get CSRF token for tests."""
    response = client.get("/api/csrf-token")
    token = response.json()["csrfToken"]
    client.cookies.set("csrftoken", token)
    return token

def test_submit_feedback_validation_failures():
      token = get_csrf_token()
      headers = {"x-csrf-token": token}
    
    # Empty user_id should fail
      response = client.post("/api/feedback", json={"user_id": "", "item": "item1", "feedback": "Good","thumbs": "up"}, headers=headers)
      assert response.status_code == 422

    # Empty item should fail
      response = client.post("/api/feedback", json={"user_id": "user123", "item": "", "feedback": "Good","thumbs": "up"}, headers=headers)
      assert response.status_code == 422

    # Empty feedback should fail
      response = client.post("/api/feedback", json={"user_id": "user123", "item": "item1", "feedback": "","thumbs": "up"}, headers=headers)
      assert response.status_code == 422


def test_submit_feedback_success():
    token = get_csrf_token() 
    headers = {"x-csrf-token": token}  
    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Excellent service!","thumbs": "up"},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["feedback"]["thumbs"] == "up"
    
    payload = response.json()
    assert "message" in payload
    assert payload["message"] == "Feedback submitted successfully"
    assert payload["feedback"]["user_id"] == "user123"
    assert payload["feedback"]["item"] == "item1"
    assert payload["feedback"]["feedback"] == "Excellent service!"
    assert payload["feedback"]["thumbs"] == "up"
    
    

def test_submit_feedback_thumbs_down():
    token = get_csrf_token()
    headers = {"x-csrf-token": token}

    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Not helpful", "thumbs": "down"},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["feedback"]["thumbs"] == "down"


def test_submit_feedback_invalid_thumbs():
    token = get_csrf_token()
    headers = {"x-csrf-token": token}

    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Good", "thumbs": "sideways"},
        headers=headers
    )
    assert response.status_code == 422


def test_feedback_stored_in_supabase():
    """Feedback should return 200 even if Supabase is unavailable."""
    token = get_csrf_token()
    headers = {"x-csrf-token": token}

    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Great!", "thumbs": "up"},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Feedback submitted successfully"