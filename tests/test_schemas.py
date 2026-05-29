import pytest
from pydantic import ValidationError
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