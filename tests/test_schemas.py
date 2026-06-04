"""
Unit tests for model schemas (pydantic validation).
Run with: pytest tests/test_schemas.py -v

Extended with comprehensive HybridWeightsSchema weight normalization
validation tests as part of issue #618.
"""
from src.model.schemas import ModelHyperparametersSchema, HybridWeightsSchema
from pydantic import ValidationError
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


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

    # ------------------------------------------------------------------
    # Original baseline tests (preserved)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Issue #618 — Extended normalization validation tests
    # ------------------------------------------------------------------

    # --- Valid normalization scenarios --------------------------------

    def test_weights_sum_exactly_one_precise(self):
        """Weights that sum to exactly 1.0 are valid and stored correctly."""
        schema = HybridWeightsSchema(alpha=0.4, beta=0.35, gamma=0.25)
        total = schema.alpha + schema.beta + schema.gamma
        assert abs(total - 1.0) < 1e-9

    def test_equal_weights_valid(self):
        """Three equal weights are accepted."""
        schema = HybridWeightsSchema(
            alpha=1 / 3, beta=1 / 3, gamma=1 / 3
        )
        total = schema.alpha + schema.beta + schema.gamma
        assert abs(total - 1.0) < 1e-9

    def test_dominant_single_weight(self):
        """One weight dominates while others are near-zero (but positive total)."""  # noqa: E501
        schema = HybridWeightsSchema(alpha=0.98, beta=0.01, gamma=0.01)
        assert schema.alpha == 0.98

    def test_weights_sum_less_than_one_accepted(self):
        """Sum < 1 is valid — the schema does not require normalisation to 1."""  # noqa: E501
        schema = HybridWeightsSchema(alpha=0.2, beta=0.2, gamma=0.2)
        assert schema.alpha + schema.beta + schema.gamma == pytest.approx(0.6)

    def test_weights_sum_greater_than_one_accepted(self):
        """Sum > 1 is valid — the schema allows un-normalised inputs."""
        schema = HybridWeightsSchema(alpha=0.9, beta=0.9, gamma=0.9)
        assert schema.alpha + schema.beta + schema.gamma == pytest.approx(2.7)

    # --- Boundary values: 0 and 1 -------------------------------------

    def test_alpha_exactly_zero_valid_when_others_positive(self):
        schema = HybridWeightsSchema(alpha=0.0, beta=0.6, gamma=0.4)
        assert schema.alpha == 0.0

    def test_beta_exactly_zero_valid_when_others_positive(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.0, gamma=0.5)
        assert schema.beta == 0.0

    def test_gamma_exactly_zero_valid_when_others_positive(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.5, gamma=0.0)
        assert schema.gamma == 0.0

    def test_two_weights_zero_single_nonzero_valid(self):
        """Only one weight is non-zero — total > 0 so it should be accepted."""
        schema = HybridWeightsSchema(alpha=1.0, beta=0.0, gamma=0.0)
        assert schema.alpha == 1.0

    def test_all_weights_at_upper_boundary(self):
        """All weights at 1.0 are individually valid (each ≤ 1.0)."""
        schema = HybridWeightsSchema(alpha=1.0, beta=1.0, gamma=1.0)
        assert schema.alpha == schema.beta == schema.gamma == 1.0

    # --- Invalid / negative values ------------------------------------

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=-0.1, beta=0.5, gamma=0.5)

    def test_negative_beta_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=0.5, beta=-0.5, gamma=0.5)

    def test_negative_gamma_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=0.5, beta=0.5, gamma=-0.1)

    def test_all_negative_weights_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=-0.3, beta=-0.3, gamma=-0.3)

    def test_alpha_above_one_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=1.1, beta=0.0, gamma=0.0)

    def test_beta_above_one_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=0.0, beta=1.01, gamma=0.0)

    def test_gamma_above_one_rejected(self):
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=1.5)

    # --- Zero-sum guard (model_validator) -----------------------------

    def test_all_zero_raises_value_error(self):
        """The model_validator must fire and raise when total == 0."""
        with pytest.raises(ValidationError) as exc_info:
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=0.0)
        assert "greater than 0" in str(exc_info.value).lower() or \
               "cumulative" in str(exc_info.value).lower() or \
               "blending" in str(exc_info.value).lower()

    def test_all_zero_error_message_content(self):
        """Validation error message should reference the normalization constraint."""  # noqa: E501
        with pytest.raises(ValidationError) as exc_info:
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=0.0)
        errors = exc_info.value.errors()
        messages = " ".join(str(e.get("msg", "")) for e in errors).lower()
        assert "0" in messages or "greater" in messages or "sum" in messages

    # --- Floating-point precision edge cases --------------------------

    def test_floating_point_near_zero_sum_rejected(self):
        """Weights that are each zero remain zero — total == 0.0, rejected."""
        with pytest.raises(ValidationError):
            HybridWeightsSchema(alpha=0.0, beta=0.0, gamma=0.0)

    def test_very_small_positive_weight_accepted(self):
        """A very small but strictly positive total should pass the validator."""  # noqa: E501
        schema = HybridWeightsSchema(alpha=1e-9, beta=0.0, gamma=0.0)
        assert schema.alpha == pytest.approx(1e-9)

    def test_floating_point_precision_sum_close_to_one(self):
        """IEEE 754 additions may not equal 1.0 exactly — schema should not care."""  # noqa: E501
        a, b, g = 0.1, 0.2, 0.7
        schema = HybridWeightsSchema(alpha=a, beta=b, gamma=g)
        total = schema.alpha + schema.beta + schema.gamma
        assert abs(total - 1.0) < 1e-9

    def test_repeating_decimal_weights(self):
        """1/3 is a repeating decimal in binary — schema must handle it cleanly."""  # noqa: E501
        schema = HybridWeightsSchema(
            alpha=round(1 / 3, 10),
            beta=round(1 / 3, 10),
            gamma=round(1 / 3, 10),
        )
        total = schema.alpha + schema.beta + schema.gamma
        assert abs(total - 1.0) < 1e-6

    def test_nan_weight_rejected(self):
        """NaN is not a valid float weight."""
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha=float("nan"), beta=0.5, gamma=0.5)

    def test_inf_weight_rejected(self):
        """Infinity exceeds the ge/le bounds and must be rejected."""
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha=float("inf"), beta=0.0, gamma=0.0)

    def test_negative_inf_weight_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha=float("-inf"), beta=0.5, gamma=0.5)

    # --- Type validation and malformed payloads -----------------------

    def test_string_alpha_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha="high", beta=0.3, gamma=0.2)

    def test_string_numeric_coerced_or_rejected(self):
        """Pydantic v2 coerces '0.5' to 0.5 for float fields by default."""
        # Either coercion succeeds or a ValidationError is raised — both are fine.  # noqa: E501
        try:
            schema = HybridWeightsSchema(alpha="0.5", beta=0.3, gamma=0.2)
            assert schema.alpha == pytest.approx(0.5)
        except (ValidationError, ValueError):
            pass  # strict mode rejection is also acceptable

    def test_none_alpha_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha=None, beta=0.5, gamma=0.5)

    def test_list_value_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha=[0.4], beta=0.3, gamma=0.2)

    def test_dict_value_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha={"value": 0.4}, beta=0.3, gamma=0.2)

    def test_boolean_weight_coerced_or_rejected(self):
        """bool is a subclass of int in Python; Pydantic may coerce True → 1.0."""  # noqa: E501
        try:
            schema = HybridWeightsSchema(alpha=True, beta=0.0, gamma=0.0)
            # If coerced, True → 1.0 which is within [0, 1]
            assert schema.alpha in (1.0, True)
        except (ValidationError, ValueError):
            pass  # rejection is also acceptable

    def test_integer_zero_weight_accepted_if_others_nonzero(self):
        """Integer 0 should be coerced to 0.0 float."""
        schema = HybridWeightsSchema(alpha=0, beta=0.5, gamma=0.5)
        assert schema.alpha == 0.0

    def test_extra_unknown_field_rejected(self):
        with pytest.raises((ValidationError, Exception)):
            HybridWeightsSchema(alpha=0.4, beta=0.35, gamma=0.25, delta=0.1)

    # --- Immutability (frozen model) ----------------------------------

    def test_alpha_immutable_after_creation(self):
        schema = HybridWeightsSchema()
        with pytest.raises(Exception):
            schema.alpha = 0.1

    def test_beta_immutable_after_creation(self):
        schema = HybridWeightsSchema()
        with pytest.raises(Exception):
            schema.beta = 0.1

    def test_gamma_immutable_after_creation(self):
        schema = HybridWeightsSchema()
        with pytest.raises(Exception):
            schema.gamma = 0.1

    # --- Determinism and stability -----------------------------------

    def test_same_inputs_produce_equal_schemas(self):
        """Instantiating twice with identical inputs yields equal objects."""
        s1 = HybridWeightsSchema(alpha=0.4, beta=0.35, gamma=0.25)
        s2 = HybridWeightsSchema(alpha=0.4, beta=0.35, gamma=0.25)
        assert s1 == s2

    def test_different_inputs_produce_unequal_schemas(self):
        s1 = HybridWeightsSchema(alpha=0.4, beta=0.35, gamma=0.25)
        s2 = HybridWeightsSchema(alpha=0.5, beta=0.3, gamma=0.2)
        assert s1 != s2

    def test_schema_hash_stable(self):
        """Frozen Pydantic models are hashable; hash must be stable."""
        s = HybridWeightsSchema(alpha=0.4, beta=0.35, gamma=0.25)
        assert hash(s) == hash(s)

    def test_repeated_instantiation_consistent(self):
        """Repeated construction with the same args always returns the same values."""  # noqa: E501
        results = [
            HybridWeightsSchema(alpha=0.3, beta=0.4, gamma=0.3)
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results)

    # --- Ratio / consistency checks -----------------------------------

    def test_alpha_greater_than_beta_preserved(self):
        schema = HybridWeightsSchema(alpha=0.6, beta=0.3, gamma=0.1)
        assert schema.alpha > schema.beta

    def test_weight_ordering_preserved(self):
        schema = HybridWeightsSchema(alpha=0.5, beta=0.3, gamma=0.2)
        assert schema.alpha > schema.beta > schema.gamma

    def test_relative_ratio_alpha_beta(self):
        """alpha:beta ratio is preserved exactly as supplied."""
        schema = HybridWeightsSchema(alpha=0.6, beta=0.3, gamma=0.1)
        assert schema.alpha / schema.beta == pytest.approx(2.0)

    # --- Parametrized valid inputs -----------------------------------

    @pytest.mark.parametrize("alpha,beta,gamma", [
        (0.4, 0.35, 0.25),    # default
        (0.5, 0.3, 0.2),      # sum == 1
        (0.1, 0.1, 0.1),      # sum < 1
        (0.9, 0.9, 0.9),      # sum > 1
        (1.0, 0.0, 0.0),      # single dominant weight
        (0.0, 1.0, 0.0),      # beta dominant
        (0.0, 0.0, 1.0),      # gamma dominant
        (1e-9, 0.0, 0.0),     # near-zero but positive total
        (1.0, 1.0, 1.0),      # all at ceiling
    ])
    def test_valid_weight_combinations_accepted(self, alpha, beta, gamma):
        schema = HybridWeightsSchema(alpha=alpha, beta=beta, gamma=gamma)
        assert schema.alpha == pytest.approx(alpha)
        assert schema.beta == pytest.approx(beta)
        assert schema.gamma == pytest.approx(gamma)

    # --- Parametrized invalid inputs ---------------------------------

    @pytest.mark.parametrize("alpha,beta,gamma", [
        (0.0, 0.0, 0.0),      # total == 0 — validator rejects
        (-0.1, 0.5, 0.5),     # negative alpha
        (0.5, -0.5, 0.5),     # negative beta
        (0.5, 0.5, -0.1),     # negative gamma
        (1.5, 0.0, 0.0),      # alpha > 1
        (0.0, 1.1, 0.0),      # beta > 1
        (0.0, 0.0, 2.0),      # gamma > 1
    ])
    def test_invalid_weight_combinations_rejected(self, alpha, beta, gamma):
        with pytest.raises((ValidationError, ValueError)):
            HybridWeightsSchema(alpha=alpha, beta=beta, gamma=gamma)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
