# File path: tests/test_xai.py
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_xai_explanation_endpoint():
    response = client.get("/api/recommendations/item123/explanation?user_id=user456")
    assert response.status_code == 200
    
    json_data = response.json()
    assert json_data["status"] == "success"
    
    percentages = json_data["data"]["breakdown_percentages"]
    # Explicitly verify math totals exactly 100%
    total_percentage = percentages["content"] + percentages["collaborative"] + percentages["sentiment"]
    assert total_percentage == 100
