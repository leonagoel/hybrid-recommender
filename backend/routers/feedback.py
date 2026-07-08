from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"]
)

feedback_store = []

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    feedback_type: str

@router.post("/")
def submit_feedback(data: FeedbackRequest):
    feedback_store.append({
        "user_id": data.user_id,
        "item_id": data.item_id,
        "feedback_type": data.feedback_type,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {"success": True}

@router.get("/user/{user_id}")
def get_user_feedback(user_id: str):
    return [
        f for f in feedback_store
        if f["user_id"] == user_id
    ]

@router.get("/stats")
def get_feedback_stats():
    stats = {
        "LIKE": 0,
        "DISLIKE": 0,
        "SAVE": 0,
        "SKIP": 0
    }

    for f in feedback_store:
        if f["feedback_type"] in stats:
            stats[f["feedback_type"]] += 1

    return stats
