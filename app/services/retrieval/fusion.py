"""Hybrid result fusion. See docs/decisions/0003-rrf-for-hybrid-fusion.md
for why Reciprocal Rank Fusion (RRF) rather than score normalization.

`FusionStrategy` is a Protocol so alternative strategies (weighted-sum
after normalization, learned fusion) can be swapped in and evaluated
without changing callers — see docs/retrieval-design.md.
"""

from typing import Protocol


class FusionStrategy(Protocol):
    def fuse(
        self,
        rankings: dict[str, list[str]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """rankings: {retriever_name: [id, ...]} each already sorted best-first.
        Returns [(id, fused_score), ...] sorted best-first."""
        ...


class ReciprocalRankFusion:
    """score(id) = sum over retrievers r containing id of
    weight_r / (k + rank_r(id)), rank_r starting at 1 for the best result.
    """

    def __init__(self, k: int = 60):
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k

    def fuse(
        self,
        rankings: dict[str, list[str]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        weights = weights or {}
        scores: dict[str, float] = {}

        for retriever, ranked_ids in rankings.items():
            weight = weights.get(retriever, 1.0)
            for rank, item_id in enumerate(ranked_ids, start=1):
                scores[item_id] = scores.get(item_id, 0.0) + weight / (self.k + rank)

        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
