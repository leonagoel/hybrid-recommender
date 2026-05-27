import pytest
from pydantic import ValidationError

from backend.main import FeedbackCreate, PurchaseCreate, RealtimeRecommendationRequest, WeightsUpdate


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (PurchaseCreate, {"user_id": "u1", "product_id": 1, "role": "admin"}),
        (FeedbackCreate, {"user_id": "u1", "item": "Alpha", "feedback": "ok", "is_admin": True}),
        (WeightsUpdate, {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "owner": "attacker"}),
        (RealtimeRecommendationRequest, {"item_title": "Alpha", "top_n": 5, "user_id": "victim"}),
    ],
)
def test_write_models_reject_extra_fields(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_purchase_model_accepts_expected_fields_only():
    purchase = PurchaseCreate.model_validate({
        "user_id": "u1",
        "product_id": 42,
        "rating": 4.5,
        "review_text": "useful",
    })

    assert purchase.user_id == "u1"
    assert purchase.product_id == 42
