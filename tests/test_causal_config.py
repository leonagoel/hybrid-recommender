"""
Unit tests specifically for CausalConfig dataclass.
Run with: pytest tests/ -v
"""
import pytest
from src.model.causal_config import CausalConfig


class TestCausalConfigSpec:

    def test_default_config_properties(self):
        cfg = CausalConfig()
        assert cfg.enabled is True
        assert cfg.blend_lambda == 0.5
        assert cfg.clip_max == 5.0
        assert cfg.score_key == 'hybrid_score'
        # Validation should succeed
        assert cfg.validate() is cfg

    def test_invalid_enabled_raises(self):
        cfg = CausalConfig(enabled="not-a-bool")
        with pytest.raises(ValueError, match="enabled must be a bool"):
            cfg.validate()

    def test_invalid_lambda_out_of_bounds(self):
        with pytest.raises(ValueError, match="blend_lambda must be in"):
            CausalConfig(blend_lambda=1.1).validate()
        with pytest.raises(ValueError, match="blend_lambda must be in"):
            CausalConfig(blend_lambda=-0.1).validate()

    def test_invalid_clip_non_positive(self):
        with pytest.raises(ValueError, match="clip_max must be positive"):
            CausalConfig(clip_max=0.0).validate()
        with pytest.raises(ValueError, match="clip_max must be positive"):
            CausalConfig(clip_max=-1.5).validate()

    def test_invalid_score_key_empty(self):
        with pytest.raises(ValueError, match="score_key must be a non-empty string"):
            CausalConfig(score_key="").validate()

    def test_presets(self):
        disabled = CausalConfig.disabled()
        assert disabled.enabled is False

        conservative = CausalConfig.conservative()
        assert conservative.enabled is True
        assert conservative.blend_lambda == 0.3
        assert conservative.clip_max == 3.0

        aggressive = CausalConfig.aggressive()
        assert aggressive.enabled is True
        assert aggressive.blend_lambda == 0.8
        assert aggressive.clip_max == 8.0

    def test_to_from_dict(self):
        original = CausalConfig(enabled=True, blend_lambda=0.75, clip_max=4.2, score_key='custom')
        d = original.to_dict()
        assert d['enabled'] is True
        assert d['blend_lambda'] == 0.75
        assert d['clip_max'] == 4.2
        assert d['score_key'] == 'custom'

        restored = CausalConfig.from_dict(d)
        assert restored.enabled is True
        assert restored.blend_lambda == 0.75
        assert restored.clip_max == 4.2
        assert restored.score_key == 'custom'

    def test_from_dict_defaults(self):
        restored = CausalConfig.from_dict({})
        assert restored.enabled is True
        assert restored.blend_lambda == 0.5
        assert restored.clip_max == 5.0
        assert restored.score_key == 'hybrid_score'

    def test_from_dict_validation_error(self):
        with pytest.raises(ValueError):
            CausalConfig.from_dict({'blend_lambda': -5.0})
