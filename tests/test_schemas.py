import pytest
from pydantic import ValidationError
from src.model.schemas import (
    RecommendRequest,
    SearchRequest,
    WeightUpdateRequest,
    RecommendResponse,
    HealthResponse,
)

# ========== Request Schemas ==========

class TestRecommendRequest:
    def test_valid_request(self):
        req = RecommendRequest(item_title="Laptop", top_n=10)
        assert req.item_title == "Laptop"
        assert req.top_n == 10

    def test_missing_item_title_raises(self):
        with pytest.raises(ValidationError):
            RecommendRequest()

    def test_top_n_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            RecommendRequest(item_title="A", top_n=0)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RecommendRequest(item_title="A", top_n=5, unknown="forbidden")

class TestSearchRequest:
    def test_valid_search(self):
        req = SearchRequest(q="phone", limit=20, offset=0)
        assert req.q == "phone"
        assert req.limit == 20

    def test_missing_q_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest()

    def test_limit_default(self):
        req = SearchRequest(q="book")
        # Update default value based on actual schema (e.g., 10)
        assert req.limit == 10

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            SearchRequest(q="a", limit=0)

class TestWeightUpdateRequest:
    def test_valid_weights(self):
        req = WeightUpdateRequest(alpha=0.5, beta=0.3, gamma=0.2)
        assert req.alpha + req.beta + req.gamma == 1.0

    def test_weights_out_of_range(self):
        with pytest.raises(ValidationError):
            WeightUpdateRequest(alpha=1.5, beta=0, gamma=0)

    def test_sum_le_zero_raises(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            WeightUpdateRequest(alpha=0, beta=0, gamma=0)

# ========== Response Schemas ==========

class TestRecommendResponse:
    def test_serialization(self):
        data = {
            "results": [{"title": "Item1", "score": 0.95}],
            "status": "success"
        }
        resp = RecommendResponse(**data)
        assert resp.results[0]["title"] == "Item1"
        # Test conversion to dict
        as_dict = resp.dict()
        assert "results" in as_dict

class TestHealthResponse:
    def test_health_response_fields(self):
        resp = HealthResponse(status="ok", model_ready=True, product_count=100)
        assert resp.status == "ok"
        assert resp.model_ready is True
        assert resp.product_count == 100
        # Test JSON serialization
        assert isinstance(resp.json(), str)
from model.schemas import HybridWeightsSchema, ModelHyperparametersSchema

class TestHybridWeightsSchema:
    def test_default_values(self):
        weights = HybridWeightsSchema()
        assert weights.alpha == 0.4
        assert weights.beta == 0.35
        assert weights.gamma == 0.25

    def test_normalization_sum_less_than_or_equal_zero(self):
        with pytest.raises(ValueError, match="cumulative summation.*must be greater than 0"):
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=0.0)
        with pytest.raises(ValueError, match="cumulative summation.*must be greater than 0"):
            HybridWeightsSchema(alpha=-0.5, beta=0.2, gamma=-0.2)

    def test_boundary_values(self):
        weights = HybridWeightsSchema(alpha=1.0, beta=0.0, gamma=0.0)
        assert weights.alpha == 1.0
        assert weights.beta == 0.0
        assert weights.gamma == 0.0

        weights = HybridWeightsSchema(alpha=0.6, beta=0.3, gamma=0.1)
        assert weights.alpha + weights.beta + weights.gamma == 1.0

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=0.5, beta=0.3, gamma=0.2, extra_field=123)

    def test_frozen_config(self):
        weights = HybridWeightsSchema()
        with pytest.raises(TypeError):
            weights.alpha = 0.9

class TestModelHyperparametersSchema:
    def test_n_factors_ge_one(self):
        params = ModelHyperparametersSchema(n_factors=50)
        assert params.n_factors == 50

        with pytest.raises(ValidationError, match="n_factors.*must be greater than or equal to 1"):
            ModelHyperparametersSchema(n_factors=0)
        with pytest.raises(ValidationError, match="n_factors.*must be greater than or equal to 1"):
            ModelHyperparametersSchema(n_factors=-5)

    def test_default_values(self):
        params = ModelHyperparametersSchema()
        assert params.n_factors == 50
        assert params.use_implicit is True

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ModelHyperparametersSchema(n_factors=50, extra_field="not allowed")

    def test_frozen_config(self):
        params = ModelHyperparametersSchema()
        with pytest.raises(TypeError):
            params.n_factors = 100
