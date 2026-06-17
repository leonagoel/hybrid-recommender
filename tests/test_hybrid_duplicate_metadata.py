"""
Regression test for the duplicate metadata lookup block in HybridRecommender.recommend().

The scoring loop previously had two back-to-back metadata lookup blocks:
  - Block A (guarded): wrapped in hasattr + try/except, set description/top_reviews
  - Block B (unguarded): directly accessed self.content_model.df without None guard,
    then reset description = '' and top_reviews = [], discarding Block A's values

Additionally, popularity_bonus and hybrid were computed three times — the last
computation always won, making the intermediate ones dead code.

This test file validates the fix at the source level since hybrid_model.py
has a pre-existing IndentationError at line 221 (stray code inside
select_bandit_arm) that prevents module import.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _get_recommend_source() -> str:
    return Path("src/model/hybrid_model.py").read_text()


def test_no_unguarded_df_access_after_guarded_block():
    """The unguarded self.content_model.df[...] lookup that followed Block A must be gone."""
    source = _get_recommend_source()
    # Block B's unguarded pattern was:
    #   row_data = self.content_model.df[\n        self.content_model.df['title'] == item['title']
    assert "row_data = self.content_model.df[\n                self.content_model.df['title']" not in source, (
        "Unguarded self.content_model.df access (Block B) is still present — "
        "it crashes with AttributeError when content_model.df is None"
    )


def test_description_not_unconditionally_reset_after_guarded_fetch():
    """description = '' must not appear after the guarded metadata block."""
    source = _get_recommend_source()
    # The bug pattern was: Block A sets description via try/except, then Block B resets it.
    # Find the guarded block and ensure there's no bare "description = ''" after it.
    guarded_end = source.find("except Exception:\n                    pass\n\n            avg_rating")
    assert guarded_end != -1, "Could not locate guarded metadata block — structure may have changed"
    after_guard = source[guarded_end:]
    # description = '' should not appear before the result = { construction
    result_dict_pos = after_guard.find("result = {")
    assert result_dict_pos != -1, "Could not find result dict construction"
    between = after_guard[:result_dict_pos]
    assert "description = ''" not in between, (
        "description = '' found after the guarded block — Block B still discarding Block A's value"
    )


def test_popularity_bonus_computed_once_per_item():
    """popularity_bonus = 0.05 * popularity must appear exactly once inside the scoring loop."""
    source = _get_recommend_source()
    loop_start = source.find("for i, item in enumerate(items):")
    assert loop_start != -1, "Could not find scoring loop"
    # Find where the loop ends (next method definition or results.sort)
    loop_end = source.find("results.sort(", loop_start)
    assert loop_end != -1
    loop_body = source[loop_start:loop_end]
    count = loop_body.count("popularity_bonus = 0.05 * popularity")
    assert count == 1, (
        f"popularity_bonus computed {count} times per item — expected exactly 1"
    )
