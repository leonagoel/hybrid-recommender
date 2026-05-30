from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

class WeightsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2

class PurchaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.@]+$")
    product_id: int = Field(..., gt=0)
    rating: float = Field(0.0, ge=0.0, le=5.0)
    review_text: str = Field("", max_length=1000)

class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.@]+$")
    item: str = Field(..., min_length=1, max_length=500)
    feedback: str = Field(..., min_length=1, max_length=2000)
    thumbs: str = Field(..., pattern=r"^(up|down)$")

class RealtimeRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_title: str
    top_n: int = 10
    explain: bool = False
    target_catalog: Optional[str] = None

class FederatedTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_factors: int = 20
    epochs: int = 5
    lr: float = 0.05
    reg: float = 0.05
