"""Retrieval-quality metrics (docs/evaluation.md), computed independently
of generation so a bad prompt can't mask good retrieval or vice versa.

Deliberately generic: a ranked list of retrieved ids + a set of expected-
relevant ids is enough to score either document-chunk retrieval (ids =
`document_id`) or graph employee-match retrieval (ids = `pg_id`) against
the same benchmark harness — see `scripts/evaluate.py`.
"""

import math
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate: float
    ndcg_at_k: float


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_retrieval_metrics(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> RetrievalMetrics:
    """`relevant_ids` is the benchmark's ground truth for one query; it must
    be non-empty — a case with no known-relevant items can't be scored for
    recall/precision and shouldn't be in the benchmark."""
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty to compute retrieval metrics")
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = retrieved_ids[:k]
    hits = [1 if rid in relevant_ids else 0 for rid in top_k]
    num_hits = sum(hits)

    recall_at_k = num_hits / len(relevant_ids)
    precision_at_k = num_hits / k
    hit_rate = 1.0 if num_hits > 0 else 0.0

    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            mrr = 1.0 / (idx + 1)
            break

    ideal_hits = [1] * min(len(relevant_ids), k) + [0] * max(0, k - len(relevant_ids))
    idcg = _dcg(ideal_hits)
    ndcg_at_k = _dcg(hits) / idcg if idcg > 0 else 0.0

    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mrr=mrr,
        hit_rate=hit_rate,
        ndcg_at_k=ndcg_at_k,
    )


def average_metrics(metrics_list: list[RetrievalMetrics]) -> RetrievalMetrics:
    n = len(metrics_list)
    if n == 0:
        raise ValueError("metrics_list must be non-empty")
    return RetrievalMetrics(
        recall_at_k=sum(m.recall_at_k for m in metrics_list) / n,
        precision_at_k=sum(m.precision_at_k for m in metrics_list) / n,
        mrr=sum(m.mrr for m in metrics_list) / n,
        hit_rate=sum(m.hit_rate for m in metrics_list) / n,
        ndcg_at_k=sum(m.ndcg_at_k for m in metrics_list) / n,
    )
