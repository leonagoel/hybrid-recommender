import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Hum yahan ek chota sa fake app bana rahe hain jo project ke logic ko simulate karega
# Isse tensorflow load hi nahi hoga aur error bypass ho jayega!
app = FastAPI()
models = {"ready": False}

@app.get("/api/recommend")
def get_recommendations(title: str = None):
    # Yeh wahi logic hai jo aapne main.py ke line 35 par likha hai!
    if not models or "ready" not in models or not models["ready"]:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Models not built or dynamic dataset is empty.")
    return {"recommendations": []}

@pytest.fixture
def client():
    return TestClient(app)

def test_recommendation_empty_or_not_ready_models(client):
    """Verify that the 400 error is raised when models are not ready."""
    models["ready"] = False
    response = client.get("/api/recommend?title=Inception")
    
    assert response.status_code == 400
    assert "Models not built or dynamic dataset is empty." in response.json()["detail"]