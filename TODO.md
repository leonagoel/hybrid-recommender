# TODO - Recommendation Explanation Feature (#1121)

- [x] Add backend support to return human-readable `explanation` for each recommended item when `explain=true` (HybridRecommender).
- [x] Update frontend recommendation cards (realtime + HTTP) to display `Reason:` under each item.
- [ ] Fix any remaining frontend JS syntax issues introduced during wiring.
- [ ] Smoke test endpoint: `/api/recommend/{title}?top_n=12&explain=true` and verify UI renders reason text.
- [ ] Run backend/frontend automated tests (pytest / npm) if available in environment.

