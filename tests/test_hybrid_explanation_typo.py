from src.model.hybrid_model import HybridRecommender


class _ContentStub:
    def explain_similarity(self, source_title, candidate_title):
        return []


def test_build_explanation_uses_round():
    model = HybridRecommender(_ContentStub(), item_df=None)
    explanation = model._build_explanation(
        source_title='Product A',
        candidate_title='Product B',
        content_score=0.7,
        collab_score=0.2,
        sentiment_score=0.1,
        popularity=0.4,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        raw_item={'raw_content': 0.6, 'raw_collab': 0.1, 'raw_sentiment': 0.0},
    )
    assert explanation['component_scores']['raw_content'] == 0.6
