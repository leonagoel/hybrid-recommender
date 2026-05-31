import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.causal_model import CausalDebiaser
from src.model.causal_config import CausalConfig


class TestCausalDebiaser:
    def test_init_default(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        debiaser = CausalDebiaser(item_df)
        assert debiaser.blend_lambda == 0.5
        assert debiaser.clip_max == 5.0

    def test_init_custom_lambda(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        debiaser = CausalDebiaser(item_df, blend_lambda=0.8)
        assert debiaser.blend_lambda == 0.8

    def test_init_custom_clip_max(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        debiaser = CausalDebiaser(item_df, clip_max=10.0)
        assert debiaser.clip_max == 10.0

    def test_init_invalid_lambda_low(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, blend_lambda=-0.1)

    def test_init_invalid_lambda_high(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, blend_lambda=1.1)

    def test_init_invalid_clip_max(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, clip_max=0)

    def test_debias_batch_empty(self):
        item_df = pd.DataFrame({'item_id': ['i1', 'i2'], 'title': ['Item 1', 'Item 2']})
        debiaser = CausalDebiaser(item_df)
        results = []
        debiased = debiaser.debias_batch(results)
        assert debiased == []

    def test_debias_batch_single_item(self):
        item_df = pd.DataFrame({'item_id': ['i1'], 'title': ['Item 1']})
        debiaser = CausalDebiaser(item_df)
        results = [{'item_id': 'i1', 'hybrid_score': 0.5}]
        debiased = debiaser.debias_batch(results)
        assert len(debiased) == 1
        assert 'causal_score' in debiased[0]

    def test_debias_batch_preserves_original(self):
        item_df = pd.DataFrame({'item_id': ['i1'], 'title': ['Item 1']})
        debiaser = CausalDebiaser(item_df)
        results = [{'item_id': 'i1', 'hybrid_score': 0.7}]
        debiased = debiaser.debias_batch(results)
        assert 'hybrid_score' in debiased[0]
        assert debiased[0]['hybrid_score'] == 0.7
