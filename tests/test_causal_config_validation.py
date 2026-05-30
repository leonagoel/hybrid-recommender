import pytest

from src.model.causal_config import CausalConfig


class TestCausalConfigValidation:

    def test_valid_default_config(self):
        cfg = CausalConfig()
        cfg.validate()

    def test_valid_custom_config(self):
        cfg = CausalConfig(
            enabled=True, blend_lambda=0.7, clip_max=4.0, score_key="custom_score"
        )
        result = cfg.validate()
        assert result.enabled is True
        assert result.blend_lambda == 0.7
        assert result.clip_max == 4.0
        assert result.score_key == "custom_score"

    def test_blend_lambda_negative_raises(self):
        cfg = CausalConfig(blend_lambda=-0.1)
        with pytest.raises(ValueError, match="blend_lambda must be in"):
            cfg.validate()

    def test_blend_lambda_above_one_raises(self):
        cfg = CausalConfig(blend_lambda=1.5)
        with pytest.raises(ValueError, match="blend_lambda must be in"):
            cfg.validate()

    def test_blend_lambda_boundary_zero(self):
        cfg = CausalConfig(blend_lambda=0.0)
        cfg.validate()

    def test_blend_lambda_boundary_one(self):
        cfg = CausalConfig(blend_lambda=1.0)
        cfg.validate()

    def test_clip_max_zero_raises(self):
        cfg = CausalConfig(clip_max=0.0)
        with pytest.raises(ValueError, match="clip_max must be positive"):
            cfg.validate()

    def test_clip_max_negative_raises(self):
        cfg = CausalConfig(clip_max=-2.0)
        with pytest.raises(ValueError, match="clip_max must be positive"):
            cfg.validate()

    def test_enabled_non_bool_raises(self):
        cfg = CausalConfig(enabled="true")
        with pytest.raises(ValueError, match="enabled must be a bool"):
            cfg.validate()

    def test_score_key_empty_raises(self):
        cfg = CausalConfig(score_key="")
        with pytest.raises(ValueError, match="score_key must be a non-empty string"):
            cfg.validate()


class TestCausalConfigFromDict:

    def test_from_dict_valid(self):
        d = {
            "enabled": True,
            "blend_lambda": 0.6,
            "clip_max": 5.0,
            "score_key": "hybrid_score",
        }
        cfg = CausalConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.blend_lambda == 0.6
        assert cfg.clip_max == 5.0
        assert cfg.score_key == "hybrid_score"

    def test_from_dict_defaults(self):
        d = {}
        cfg = CausalConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.blend_lambda == 0.5
        assert cfg.clip_max == 5.0
        assert cfg.score_key == "hybrid_score"

    def test_from_dict_invalid_blend_lambda(self):
        d = {"blend_lambda": -0.5}
        with pytest.raises(ValueError):
            CausalConfig.from_dict(d)

    def test_from_dict_invalid_clip_max(self):
        d = {"clip_max": 0}
        with pytest.raises(ValueError):
            CausalConfig.from_dict(d)

    def test_from_dict_invalid_score_key(self):
        d = {"score_key": ""}
        with pytest.raises(ValueError):
            CausalConfig.from_dict(d)


class TestCausalConfigToDict:

    def test_to_dict(self):
        cfg = CausalConfig(
            enabled=True, blend_lambda=0.5, clip_max=5.0, score_key="hybrid_score"
        )
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["blend_lambda"] == 0.5
        assert d["clip_max"] == 5.0
        assert d["score_key"] == "hybrid_score"


class TestCausalConfigPresets:

    def test_disabled_preset(self):
        cfg = CausalConfig.disabled()
        assert cfg.enabled is False
        assert cfg.blend_lambda == 0.5
        assert cfg.clip_max == 5.0

    def test_conservative_preset(self):
        cfg = CausalConfig.conservative()
        assert cfg.enabled is True
        assert cfg.blend_lambda == 0.3
        assert cfg.clip_max == 3.0

    def test_aggressive_preset(self):
        cfg = CausalConfig.aggressive()
        assert cfg.enabled is True
        assert cfg.blend_lambda == 0.8
        assert cfg.clip_max == 8.0
