import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.schemas import ModelHyperparametersSchema, HybridWeightsSchema


class TestModelHyperparametersSchema:
    def test_init_defaults(self):
        schema = ModelHyperparametersSchema()
        assert schema.n_factors == 50
        assert schema.use_implicit is True

    def test_init_custom(self):
        schema = ModelHyperparametersSchema(n_factors=100, use_implicit=False)
        assert schema.n_factors == 100
        assert schema.use_implicit is False

    def test_n_factors_must_be_positive(self):
        with pytest.raises(ValueError):
            ModelHyperparametersSchema(n_factors=0)

    def test_n_factors_must_be_integer(self):
        with pytest.raises(ValueError):
            ModelHyperparametersSchema(n_factors=10.5)

    def test_frozen_config(self):
        schema = ModelHyperparametersSchema()
        with pytest.raises(Exception):
            schema.n_factors = 100


class TestHybridWeightsSchema:
    def test_init_defaults(self):
        schema = HybridWeightsSchema()
        assert schema.alpha == 0.4
        assert schema.beta == 0.35
        assert schema.gamma == 0.25

    def test_init_custom(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.3, gamma=0.2)
        assert schema.alpha == 0.5
        assert schema.beta == 0.3
        assert schema.gamma == 0.2

    def test_weights_must_be_nonnegative(self):
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha=-0.1, beta=0.5, gamma=0.6)

    def test_weights_must_be_at_most_one(self):
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha=1.5, beta=0.0, gamma=0.0)

    def test_validate_weights_normalization_zero_total(self):
        with pytest.raises(ValueError):
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=0.0)

    def test_validate_weights_normalization_valid(self):
        schema = HybridWeightsSchema(alpha=0.33, beta=0.33, gamma=0.34)
        assert schema.alpha + schema.beta + schema.gamma > 0

    def test_frozen_config(self):
        schema = HybridWeightsSchema()
        with pytest.raises(Exception):
            schema.alpha = 0.5
