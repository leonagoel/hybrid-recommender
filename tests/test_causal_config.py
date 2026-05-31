import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.causal_config import CausalConfig


class TestCausalConfig:
    def test_init_defaults(self):
        cfg = CausalConfig()
        assert cfg.enabled is True
        assert cfg.blend_lambda == 0.5
        assert cfg.clip_max == 5.0
        assert cfg.score_key == 'hybrid_score'

    def test_init_custom_values(self):
        cfg = CausalConfig(enabled=False, blend_lambda=0.8, clip_max=8.0, score_key='custom_score')
        assert cfg.enabled is False
        assert cfg.blend_lambda == 0.8
        assert cfg.clip_max == 8.0
        assert cfg.score_key == 'custom_score'

    def test_validate_valid(self):
        cfg = CausalConfig()
        result = cfg.validate()
        assert result is cfg

    def test_validate_blend_lambda_zero(self):
        cfg = CausalConfig(blend_lambda=0.0)
        result = cfg.validate()
        assert result is cfg

    def test_validate_blend_lambda_one(self):
        cfg = CausalConfig(blend_lambda=1.0)
        result = cfg.validate()
        assert result is cfg

    def test_validate_blend_lambda_invalid_low(self):
        cfg = CausalConfig(blend_lambda=-0.1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_blend_lambda_invalid_high(self):
        cfg = CausalConfig(blend_lambda=1.1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_clip_max_invalid(self):
        cfg = CausalConfig(clip_max=0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_clip_max_negative(self):
        cfg = CausalConfig(clip_max=-1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_enabled_invalid_type(self):
        cfg = CausalConfig(enabled="true")
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_score_key_empty(self):
        cfg = CausalConfig(score_key="")
        with pytest.raises(ValueError):
            cfg.validate()

    def test_to_dict(self):
        cfg = CausalConfig(enabled=False, blend_lambda=0.8, clip_max=8.0, score_key='test')
        d = cfg.to_dict()
        assert d['enabled'] is False
        assert d['blend_lambda'] == 0.8
        assert d['clip_max'] == 8.0
        assert d['score_key'] == 'test'

    def test_from_dict(self):
        d = {'enabled': False, 'blend_lambda': 0.8, 'clip_max': 8.0, 'score_key': 'test'}
        cfg = CausalConfig.from_dict(d)
        assert cfg.enabled is False
        assert cfg.blend_lambda == 0.8
        assert cfg.clip_max == 8.0
        assert cfg.score_key == 'test'

    def test_repr(self):
        cfg = CausalConfig()
        r = repr(cfg)
        assert 'CausalConfig' in r
        assert 'blend_lambda' in r
