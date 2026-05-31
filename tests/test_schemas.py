"""
Unit tests for model schemas (pydantic validation).
Run with: pytest tests/test_schemas.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.schemas import ModelHyperparametersSchema, HybridWeightsSchema


class TestModelHyperparametersSchema:
    def test_default_values(self):
        schema = ModelHyperparametersSchema()
        assert schema.n_factors == 50
        assert schema.use_implicit is True

    def test_custom_values(self):
        schema = ModelHyperparametersSchema(n_factors=100, use_implicit=False)
        assert schema.n_factors == 100
        assert schema.use_implicit is False

    def test_n_factors_must_be_positive(self):
        with pytest.raises(ValueError):
            ModelHyperparametersSchema(n_factors=0)

    def test_n_factors_must_be_integer(self):
        with pytest.raises(ValueError):
            ModelHyperparametersSchema(n_factors=-1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ModelHyperparametersSchema(unknown_field=123)

    def test_n_factors_boundary(self):
        schema = ModelHyperparametersSchema(n_factors=1)
        assert schema.n_factors == 1


class TestHybridWeightsSchema:
    def test_default_values(self):
        schema = HybridWeightsSchema()
        assert schema.alpha == 0.4
        assert schema.beta == 0.35
        assert schema.gamma == 0.25

    def test_custom_values(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.3, gamma=0.2)
        assert schema.alpha == 0.5
        assert schema.beta == 0.3
        assert schema.gamma == 0.2

    def test_weights_must_be_between_0_and_1(self):
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha=1.5)
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha=-0.1)

    def test_weights_cannot_all_be_zero(self):
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=0.0)

    def test_weights_can_sum_to_one(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.3, gamma=0.2)
        assert schema.alpha + schema.beta + schema.gamma == 1.0

    def test_weights_can_sum_to_less_than_one(self):
        schema = HybridWeightsSchema(alpha=0.1, beta=0.1, gamma=0.1)
        assert abs(schema.alpha + schema.beta + schema.gamma - 0.3) < 0.001

    def test_weights_can_sum_to_greater_than_one(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.5, gamma=0.5)
        assert schema.alpha + schema.beta + schema.gamma == 1.5

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            HybridWeightsSchema(unknown_field=0.5)

    def test_frozen_schema_cannot_be_modified(self):
        schema = HybridWeightsSchema()
        with pytest.raises(Exception):
            schema.alpha = 0.9

    def test_boundary_zero_to_one(self):
        schema = HybridWeightsSchema(alpha=0.0, beta=0.5, gamma=0.5)
        assert schema.alpha == 0.0

    def test_boundary_precision_and_invalid_types(self):
        # Precise micro floating boundaries
        schema = HybridWeightsSchema(alpha=0.0001, beta=0.9999, gamma=0.0)
        assert schema.alpha == 0.0001
        assert schema.beta == 0.9999

        # Rejection of invalid non-numeric strings
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha="invalid_string")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
