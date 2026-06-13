# TODO - Recommendation Explanation Feature (#1121 / #1352)

- [ ] Repair/replace `src/model/hybrid_model.py` (current merge-conflict artifacts) with a compilable implementation.
- [ ] Ensure `HybridRecommender.recommend(..., explain=True)` adds `explanation` field (short human-readable string) for each recommendation.
- [ ] Ensure explanation strongest-component logic aligns with hybrid scoring weights.
- [ ] Verify backend `/api/recommend` passes `explain` through and frontend `frontend/app.js` renders `r.explanation`.
- [ ] Run test suite (`pytest`) and fix any failures.

