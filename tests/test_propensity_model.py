import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.propensity_model import PropensityModel


class TestPropensityModel:
    def test_init_with_dataframe(self):
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i1'],
            'rating': [5, 3, 4]
        })
        model = PropensityModel(df)
        assert model is not None

    def test_all_scores_returns_dict(self):
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i1'],
            'rating': [5, 3, 4]
        })
        model = PropensityModel(df)
        scores = model.all_scores()
        assert isinstance(scores, dict)
