from fastapi.testclient import TestClient
from backend import main

client = TestClient(main.app)


def test_submit_feedback_validation_failures():
    # Empty user_id should fail
    response = client.post("/api/feedback", json={"user_id": "", "item": "item1", "feedback": "Good"})
    assert response.status_code == 422

    # Empty item should fail
    response = client.post("/api/feedback", json={"user_id": "user123", "item": "", "feedback": "Good"})
    assert response.status_code == 422

    # Empty feedback should fail
    response = client.post("/api/feedback", json={"user_id": "user123", "item": "item1", "feedback": ""})
    assert response.status_code == 422


def test_submit_feedback_success():
    response = client.post(
        "/api/feedback",
        json={"user_id": "user123", "item": "item1", "feedback": "Excellent service!"}
    )
    assert response.status_code == 200
    
    payload = response.json()
    assert "message" in payload
    assert payload["message"] == "Feedback submitted successfully"
    assert payload["feedback"]["user_id"] == "user123"
    assert payload["feedback"]["item"] == "item1"
    assert payload["feedback"]["feedback"] == "Excellent service!"
