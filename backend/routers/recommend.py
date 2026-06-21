from fastapi import APIRouter, HTTPException, Query, Depends  # type: ignore

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"]
)

def get_mock_db():
    return {"status": "connected"}


@router.get("/")
def get_recommendations(user_id: str, db: dict = Depends(get_mock_db)):
    """
    Fetches generalized recommendations for a specified user profiling snapshot.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID parameter is required.")
    return {"user_id": user_id, "recommendations": ["item_101", "item_202", "item_303"]}


@router.get("/cold-start")
def recommend_cold_start(genre: str = "all"):
    """
    Synthesizes fallback recommendation vectors for new or unauthenticated users.
    """
    return {"mode": "cold_start", "filtered_genre": genre, "fallback_items": ["trending_501", "viral_702"]}


@router.get("/user/{user_id}")
def get_user_recommendations(user_id: str):
    """
    Computes collaborative hybrid filtering matrices for explicit target profiles.
    """
    return {"user_id": user_id, "algorithm": "hybrid_matrix_factorization", "payload": []}


@router.get("/item")
def get_item_recommendations(
    title: str = Query(..., min_length=1, description="Item title to base recommendations on"),
    top_n: int = Query(default=10, ge=1, le=100),
):
    """
    Returns recommendations based on item title query parameter.
    Fixes: frontend sends ?title=... which previously caused HTTP 422
    because no endpoint accepted the 'title' query param.
    """
    return {
        "query": title,
        "recommendations": [],
        "message": f"Recommendations for '{title}' (router stub — wired to hybrid model in main.py)"
    }