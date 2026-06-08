from collections import Counter

class MultiObjectiveRanker:

    def __init__(
        self,
        relevance_weight=0.50,
        diversity_weight=0.15,
        novelty_weight=0.10,
        freshness_weight=0.15,
        coverage_weight=0.10,
    ):
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.freshness_weight = freshness_weight
        self.coverage_weight = coverage_weight

    def rerank(self, recommendations):

        if not recommendations:
            return recommendations

        category_counts = Counter(
            r.get("category", "unknown")
            for r in recommendations
        )

        max_count = max(category_counts.values())

        for rec in recommendations:

            relevance = rec.get("hybrid_score", 0)

            category = rec.get("category", "unknown")

            diversity = (
                1 - category_counts[category] / max_count
            )

            novelty = 1 - min(
                rec.get("rating", 0) / 5.0,
                1.0
            )

            freshness = 0.5

            coverage = (
                1.0
                if category_counts[category] <= 2
                else 0.5
            )

            final_score = (
                self.relevance_weight * relevance
                + self.diversity_weight * diversity
                + self.novelty_weight * novelty
                + self.freshness_weight * freshness
                + self.coverage_weight * coverage
            )

            rec["multi_objective_score"] = round(
                final_score,
                4
            )

        recommendations.sort(
            key=lambda x: x["multi_objective_score"],
            reverse=True,
        )

        return recommendations