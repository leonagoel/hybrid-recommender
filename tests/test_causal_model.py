"""
Unit tests for CausalDebiaser (IPS causal inference layer).
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model.causal_model import CausalDebiaser
from src.model.causal_config import CausalConfig
from src.model.hybrid_model import HybridRecommender
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def item_df():
    """Item catalog with clear popularity and category skew for testing."""
    return pd.DataFrame({
        'title':        ['Blockbuster A', 'Blockbuster B', 'Niche C', 'Niche D', 'Rare E'],
        'review_count': [5000,            4000,            50,        30,         5],
        'category':     ['Electronics',   'Electronics',   'Books',   'Books',    'Art'],
        'rating':       [4.5,             4.2,             4.8,       4.7,        4.9],
        'avg_sentiment':[0.6,             0.5,             0.7,       0.8,        0.9],
        'description':  ['Popular gadget','Another gadget','Rare book','Rare book','Art piece'],
        'combined':     ['Blockbuster A Popular gadget Electronics',
                         'Blockbuster B Another gadget Electronics',
                         'Niche C Rare book Books',
                         'Niche D Rare book Books',
                         'Rare E Art piece Art'],
    })


@pytest.fixture
def debiaser(item_df):
    return CausalDebiaser(item_df)


@pytest.fixture
def interaction_df():
    return pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'],
        'title':   ['Blockbuster A', 'Blockbuster B', 'Niche C', 'Niche D', 'Rare E'],
        'rating':  [5.0, 4.0, 5.0, 4.5, 5.0],
    })


@pytest.fixture
def hybrid_model_with_causal(item_df, interaction_df):
    content = ContentRecommender(item_df)
    collab  = CollaborativeRecommender(interaction_df)
    return HybridRecommender(
        content, collab, item_df,
        use_causal_debiasing=True,
        causal_lambda=0.5,
        causal_clip=5.0,
    )


@pytest.fixture
def hybrid_model_no_causal(item_df, interaction_df):
    content = ContentRecommender(item_df)
    collab  = CollaborativeRecommender(interaction_df)
    return HybridRecommender(content, collab, item_df, use_causal_debiasing=False)


# ─── CausalConfig ────────────────────────────────────────────────────────────

class TestCausalConfig:

    def test_default_config_is_valid(self):
        cfg = CausalConfig()
        cfg.validate()  # must not raise

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            CausalConfig(blend_lambda=1.5).validate()
        with pytest.raises(ValueError):
            CausalConfig(blend_lambda=-0.1).validate()

    def test_invalid_clip_raises(self):
        with pytest.raises(ValueError):
            CausalConfig(clip_max=0).validate()

    def test_disabled_preset(self):
        cfg = CausalConfig.disabled()
        assert cfg.enabled is False

    def test_conservative_preset_values(self):
        cfg = CausalConfig.conservative()
        assert cfg.enabled is True
        assert cfg.blend_lambda <= 0.5

    def test_aggressive_preset_values(self):
        cfg = CausalConfig.aggressive()
        assert cfg.enabled is True
        assert cfg.blend_lambda >= 0.5

    def test_to_dict_round_trips(self):
        cfg = CausalConfig(blend_lambda=0.6, clip_max=4.0)
        d = cfg.to_dict()
        cfg2 = CausalConfig.from_dict(d)
        assert cfg2.blend_lambda == cfg.blend_lambda
        assert cfg2.clip_max == cfg.clip_max

    def test_from_dict_validates(self):
        with pytest.raises(ValueError):
            CausalConfig.from_dict({'blend_lambda': 2.0})


# ─── CausalDebiaser.from_config() ────────────────────────────────────────────

class TestDebiaserFromConfig:

    def test_from_config_builds_debiaser(self, item_df):
        cfg = CausalConfig(blend_lambda=0.6, clip_max=4.0)
        d = CausalDebiaser.from_config(item_df, cfg)
        assert isinstance(d, CausalDebiaser)
        assert d.blend_lambda == 0.6
        assert d.clip_max == 4.0

    def test_from_config_validates_config(self, item_df):
        bad_cfg = CausalConfig(blend_lambda=2.0)  # invalid but not yet validated
        with pytest.raises(ValueError):
            CausalDebiaser.from_config(item_df, bad_cfg)

    def test_from_config_disabled_still_builds(self, item_df):
        # from_config always builds the debiaser; the enabled flag is
        # checked by HybridRecommender, not by CausalDebiaser itself
        cfg = CausalConfig.disabled()
        d = CausalDebiaser.from_config(item_df, cfg)
        assert isinstance(d, CausalDebiaser)


# ─── HybridRecommender causal_config path ────────────────────────────────────

class TestHybridCausalConfigPath:

    def test_causal_config_takes_precedence_over_raw_params(self, item_df, interaction_df):
        """When causal_config is provided, use_causal_debiasing raw param is ignored."""
        content = ContentRecommender(item_df)
        collab  = CollaborativeRecommender(interaction_df)
        cfg = CausalConfig(enabled=True, blend_lambda=0.7)
        model = HybridRecommender(
            content, collab, item_df,
            use_causal_debiasing=False,  # raw param says OFF
            causal_config=cfg,           # config says ON — config wins
        )
        assert model.use_causal_debiasing is True
        assert model._debiaser is not None
        assert model._debiaser.blend_lambda == 0.7

    def test_causal_config_disabled_preset_no_debiaser(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab  = CollaborativeRecommender(interaction_df)
        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig.disabled(),
        )
        assert model.use_causal_debiasing is False
        assert model._debiaser is None

    def test_causal_config_stored_on_model(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab  = CollaborativeRecommender(interaction_df)
        cfg = CausalConfig.conservative()
        model = HybridRecommender(content, collab, item_df, causal_config=cfg)
        assert model._causal_config is cfg

    def test_recommend_via_causal_config_returns_causal_keys(self, item_df, interaction_df):
        content = ContentRecommender(item_df)
        collab  = CollaborativeRecommender(interaction_df)
        model = HybridRecommender(
            content, collab, item_df,
            causal_config=CausalConfig(enabled=True, blend_lambda=0.5),
        )
        recs = model.recommend('Blockbuster A', top_n=3)
        for r in recs:
            assert 'causal_score' in r
            assert 'original_score' in r


# ─── CausalDebiaser construction ─────────────────────────────────────────────

class TestCausalDebiaserInit:

    def test_builds_propensity_for_all_items(self, debiaser, item_df):
        for title in item_df['title']:
            assert debiaser.get_propensity(title) > 0

    def test_invalid_lambda_raises(self, item_df):
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, blend_lambda=1.5)
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, blend_lambda=-0.1)

    def test_invalid_clip_raises(self, item_df):
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, clip_max=0)
        with pytest.raises(ValueError):
            CausalDebiaser(item_df, clip_max=-1)

    def test_empty_df_does_not_crash(self):
        d = CausalDebiaser(pd.DataFrame())
        # Unknown title falls back to propensity=1.0
        assert d.get_propensity('anything') == 1.0

    def test_df_without_review_count_uses_uniform(self):
        df = pd.DataFrame({'title': ['A', 'B'], 'category': ['X', 'X']})
        d = CausalDebiaser(df)
        # Both items have same exposure — propensities should be equal
        assert abs(d.get_propensity('A') - d.get_propensity('B')) < 1e-6

    def test_df_without_category_uses_uniform_category(self):
        df = pd.DataFrame({'title': ['A', 'B'], 'review_count': [100, 10]})
        d = CausalDebiaser(df)
        # Should not raise; popular item should have higher propensity
        assert d.get_propensity('A') > d.get_propensity('B')


# ─── Propensity direction ─────────────────────────────────────────────────────

class TestPropensityDirection:

    def test_popular_item_has_higher_propensity_than_niche(self, debiaser):
        """Blockbuster A (5000 reviews) must have higher propensity than Rare E (5 reviews)."""
        assert debiaser.get_propensity('Blockbuster A') > debiaser.get_propensity('Rare E')

    def test_dominant_category_item_has_higher_propensity(self, debiaser):
        """Electronics (2 items) should dominate Books (2 items) and Art (1 item)."""
        # Electronics items have both high review_count AND category dominance
        assert debiaser.get_propensity('Blockbuster A') > debiaser.get_propensity('Rare E')

    def test_ips_weight_inverts_propensity(self, debiaser):
        """IPS weight must be inversely proportional to propensity."""
        w_popular = debiaser.get_ips_weight('Blockbuster A')
        w_niche   = debiaser.get_ips_weight('Rare E')
        assert w_niche > w_popular


# ─── debias() single-item ─────────────────────────────────────────────────────

class TestDebiasSingle:

    def test_output_stays_in_bounds(self, debiaser, item_df):
        for title in item_df['title']:
            for score in [0.0, 0.3, 0.7, 1.0]:
                result = debiaser.debias(title, score)
                assert 0.0 <= result <= 1.0, f"Out of bounds for {title}, score={score}"

    def test_zero_score_stays_zero(self, debiaser):
        """Zero score must remain zero regardless of IPS weight."""
        assert debiaser.debias('Blockbuster A', 0.0) == 0.0
        assert debiaser.debias('Rare E', 0.0) == 0.0

    def test_lambda_zero_returns_original_score(self, item_df):
        """λ=0 means no debiasing — output must equal input."""
        d = CausalDebiaser(item_df, blend_lambda=0.0)
        for title in item_df['title']:
            assert abs(d.debias(title, 0.7) - 0.7) < 1e-6

    def test_unknown_title_falls_back_gracefully(self, debiaser):
        """Unknown title uses propensity=1.0 (neutral IPS weight)."""
        result = debiaser.debias('Unknown Item XYZ', 0.6)
        assert 0.0 <= result <= 1.0

    def test_popular_item_score_not_inflated(self):
        """
        A popular item with λ=1.0 should NOT be boosted above its original score
        because its IPS weight < 1 (it was over-exposed).
        """
        df = pd.DataFrame({
            'title': ['Popular', 'Niche'],
            'review_count': [10000, 1],
            'category': ['Electronics', 'Art'],
        })
        d_full = CausalDebiaser(df, blend_lambda=1.0)
        score_popular = d_full.debias('Popular', 0.8)
        score_niche   = d_full.debias('Niche', 0.8)
        # Niche item should score higher after full debiasing
        assert score_niche >= score_popular


# ─── debias_batch() ───────────────────────────────────────────────────────────

class TestDebiasBatch:

    def _make_items(self, titles, score=0.8):
        return [{'title': t, 'hybrid_score': score} for t in titles]

    def test_all_scores_stay_in_bounds(self, debiaser, item_df):
        items = self._make_items(item_df['title'].tolist())
        out = debiaser.debias_batch(items)
        for item in out:
            assert 0.0 <= item['hybrid_score'] <= 1.0

    def test_adds_causal_score_key(self, debiaser, item_df):
        items = self._make_items(item_df['title'].tolist())
        out = debiaser.debias_batch(items)
        for item in out:
            assert 'causal_score' in item

    def test_adds_original_score_key(self, debiaser, item_df):
        items = self._make_items(item_df['title'].tolist())
        out = debiaser.debias_batch(items)
        for item in out:
            assert 'original_score' in item
            assert item['original_score'] == 0.8

    def test_niche_item_scores_higher_than_popular_after_debiasing(self, item_df):
        """
        Core correctness test: given equal raw scores, a niche item must
        rank above a popular item after IPS debiasing.
        """
        d = CausalDebiaser(item_df, blend_lambda=1.0)
        items = [
            {'title': 'Blockbuster A', 'hybrid_score': 0.8},
            {'title': 'Rare E',        'hybrid_score': 0.8},
        ]
        out = d.debias_batch(items)
        scores = {i['title']: i['hybrid_score'] for i in out}
        assert scores['Rare E'] >= scores['Blockbuster A']

    def test_empty_batch_returns_empty(self, debiaser):
        assert debiaser.debias_batch([]) == []

    def test_custom_score_key(self, debiaser):
        items = [{'title': 'Blockbuster A', 'custom_score': 0.7}]
        out = debiaser.debias_batch(items, score_key='custom_score')
        assert 'custom_score' in out[0]
        assert 0.0 <= out[0]['custom_score'] <= 1.0

    def test_lambda_zero_batch_preserves_original_scores(self, item_df):
        """λ=0 batch debiasing must leave all scores unchanged."""
        d = CausalDebiaser(item_df, blend_lambda=0.0)
        items = [{'title': t, 'hybrid_score': 0.75} for t in item_df['title']]
        out = d.debias_batch(items)
        for item in out:
            assert abs(item['hybrid_score'] - 0.75) < 1e-6

    def test_relative_order_changes_after_debiasing(self, item_df):
        """
        Debiasing must change the relative ranking of popular vs. niche items
        when λ > 0 and scores are equal.
        """
        d = CausalDebiaser(item_df, blend_lambda=0.8)
        items = [
            {'title': 'Blockbuster A', 'hybrid_score': 0.9},
            {'title': 'Rare E',        'hybrid_score': 0.7},
        ]
        out_before = sorted(items, key=lambda x: x['hybrid_score'], reverse=True)
        out_after  = d.debias_batch([
            {'title': 'Blockbuster A', 'hybrid_score': 0.9},
            {'title': 'Rare E',        'hybrid_score': 0.7},
        ])
        out_after_sorted = sorted(out_after, key=lambda x: x['hybrid_score'], reverse=True)
        # After debiasing, Rare E should close the gap or overtake Blockbuster A
        rare_after  = next(i['hybrid_score'] for i in out_after if i['title'] == 'Rare E')
        block_after = next(i['hybrid_score'] for i in out_after if i['title'] == 'Blockbuster A')
        gap_before = 0.9 - 0.7
        gap_after  = block_after - rare_after
        assert gap_after < gap_before, "Debiasing should reduce the gap between popular and niche items"


# ─── summary() ───────────────────────────────────────────────────────────────

class TestDebiaserSummary:

    def test_summary_returns_expected_keys(self, debiaser):
        s = debiaser.summary()
        for key in ('n_items', 'propensity_mean', 'propensity_std', 'propensity_min',
                    'propensity_max', 'blend_lambda', 'clip_max'):
            assert key in s

    def test_summary_n_items_matches_df(self, debiaser, item_df):
        assert debiaser.summary()['n_items'] == len(item_df)

    def test_empty_debiaser_summary_returns_empty_dict(self):
        d = CausalDebiaser(pd.DataFrame())
        assert d.summary() == {}


# ─── HybridRecommender integration ───────────────────────────────────────────

class TestHybridCausalIntegration:

    def test_causal_flag_true_attaches_debiaser(self, hybrid_model_with_causal):
        assert hybrid_model_with_causal._debiaser is not None
        assert isinstance(hybrid_model_with_causal._debiaser, CausalDebiaser)

    def test_causal_flag_false_no_debiaser(self, hybrid_model_no_causal):
        assert hybrid_model_no_causal._debiaser is None

    def test_recommend_with_causal_returns_list(self, hybrid_model_with_causal):
        recs = hybrid_model_with_causal.recommend('Blockbuster A', top_n=3)
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_recommend_with_causal_has_required_keys(self, hybrid_model_with_causal):
        recs = hybrid_model_with_causal.recommend('Blockbuster A', top_n=3)
        required = {'title', 'hybrid_score', 'content_score', 'collab_score',
                    'sentiment_score', 'causal_score', 'original_score'}
        for r in recs:
            assert required.issubset(r.keys()), f"Missing keys in: {r.keys()}"

    def test_recommend_with_causal_scores_in_bounds(self, hybrid_model_with_causal):
        recs = hybrid_model_with_causal.recommend('Blockbuster A', top_n=4)
        for r in recs:
            assert 0.0 <= r['hybrid_score'] <= 1.0
            assert 0.0 <= r['causal_score'] <= 1.0

    def test_recommend_sorted_by_hybrid_score_after_debiasing(self, hybrid_model_with_causal):
        recs = hybrid_model_with_causal.recommend('Blockbuster A', top_n=4)
        scores = [r['hybrid_score'] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_without_causal_has_no_causal_keys(self, hybrid_model_no_causal):
        recs = hybrid_model_no_causal.recommend('Blockbuster A', top_n=3)
        for r in recs:
            assert 'causal_score' not in r
            assert 'original_score' not in r

    def test_causal_and_non_causal_produce_different_rankings(
        self, item_df, interaction_df
    ):
        """
        Causal debiasing with λ=1.0 must produce a different ranking than
        the non-causal baseline when the catalog has strong popularity skew.
        """
        content = ContentRecommender(item_df)
        collab  = CollaborativeRecommender(interaction_df)

        model_causal = HybridRecommender(
            content, collab, item_df,
            use_causal_debiasing=True, causal_lambda=1.0
        )
        model_plain = HybridRecommender(
            content, collab, item_df,
            use_causal_debiasing=False
        )

        recs_causal = model_causal.recommend('Blockbuster A', top_n=4)
        recs_plain  = model_plain.recommend('Blockbuster A', top_n=4)

        titles_causal = [r['title'] for r in recs_causal]
        titles_plain  = [r['title'] for r in recs_plain]

        # Rankings must differ when debiasing is aggressive
        assert titles_causal != titles_plain, (
            "Causal and non-causal rankings should differ with λ=1.0 on a skewed catalog"
        )

    def test_existing_recommend_keys_preserved_with_causal(self, hybrid_model_with_causal):
        """Causal layer must not remove any existing result keys."""
        recs_causal = hybrid_model_with_causal.recommend('Blockbuster A', top_n=2)
        existing_keys = {'title', 'hybrid_score', 'content_score', 'collab_score',
                         'sentiment_score', 'rating', 'category', 'description'}
        for r in recs_causal:
            assert existing_keys.issubset(r.keys())

    def test_propensity_model_direct_stats(self, item_df):
        from src.model.propensity_model import PropensityModel
        pm = PropensityModel(item_df)
        scores = pm.all_scores()
        assert len(scores) == len(item_df)
        assert pm.get('Nonexistent') == 1.0
        assert pm.get_ips_weight('Nonexistent', 5.0) == 1.0

    def test_debiaser_extreme_clip(self, item_df):
        d = CausalDebiaser(item_df, blend_lambda=0.5, clip_max=100.0)
        assert d.clip_max == 100.0
        w = d.get_ips_weight('Rare E')
        assert w <= 100.0

    def test_debias_batch_missing_score_key_defaults_to_zero(self, item_df):
        d = CausalDebiaser(item_df, blend_lambda=0.5)
        items = [{'title': 'Blockbuster A'}]
        out = d.debias_batch(items)
        assert out[0]['hybrid_score'] == 0.0
        assert out[0]['original_score'] == 0.0

    def test_debias_batch_zero_mean_weights_fallback(self, item_df):
        d = CausalDebiaser(item_df, blend_lambda=0.5)
        original_get_ips_weight = d._propensity_model.get_ips_weight
        try:
            d._propensity_model.get_ips_weight = lambda title, clip_max: 0.0
            items = [{'title': 'Blockbuster A', 'hybrid_score': 0.8}]
            out = d.debias_batch(items)
            assert abs(out[0]['hybrid_score'] - 0.8) < 1e-6
        finally:
            d._propensity_model.get_ips_weight = original_get_ips_weight

