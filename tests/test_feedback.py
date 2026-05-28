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


def test_submit_feedback_success():
    
    # Test: Valid feedback with thumbs up should return 200.
   
    token = get_csrf_token()
    headers = {"x-csrf-token": token}
    
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
    assert payload["feedback"]["thumbs"] == "up"

def test_submit_feedback_thumbs_down():
     
   # Test: Valid feedback with thumbs down should return 200.
    
    token = get_csrf_token()
    headers = {"x-csrf-token": token}
    
    
    response = client.post(
        "/api/feedback",
        json={
            "user_id": "user123",
            "item": "item1", 
            "feedback": "Not helpful",
            "thumbs": "down"
        }
    )
    assert response.status_code == 200
    assert response.json()["feedback"]["thumbs"] == "down"

def test_submit_feedback_invalid_thumbs():
    
   # Test: Invalid thumbs value should return 422. only up and down is allowed .
   
    token = get_csrf_token()
    headers = {"x-csrf-token": token}
    
    
    response = client.post(
        "/api/feedback",
        json={
            "user_id": "user123",
            "item": "item1",
            "feedback": "Good",
            "thumbs": "sideways"  # invalid!
        }
    )
    assert response.status_code == 422
    
