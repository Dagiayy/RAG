import pytest

from app.services.evaluation.retrieval_metrics import average_metrics, compute_retrieval_metrics


def test_perfect_retrieval_scores_1_on_everything():
    metrics = compute_retrieval_metrics(["a", "b", "c"], {"a", "b", "c"}, k=3)
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.hit_rate == 1.0
    assert metrics.ndcg_at_k == 1.0


def test_no_relevant_items_retrieved_scores_0():
    metrics = compute_retrieval_metrics(["x", "y", "z"], {"a", "b"}, k=3)
    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.mrr == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.ndcg_at_k == 0.0


def test_partial_recall_and_precision():
    # 1 of 2 relevant items retrieved, within top 4 of 4 returned
    metrics = compute_retrieval_metrics(["x", "a", "y", "z"], {"a", "b"}, k=4)
    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 0.25
    assert metrics.hit_rate == 1.0


def test_mrr_rewards_earlier_rank():
    early = compute_retrieval_metrics(["a", "x", "y"], {"a"}, k=3)
    late = compute_retrieval_metrics(["x", "y", "a"], {"a"}, k=3)
    assert early.mrr == 1.0
    assert late.mrr == pytest.approx(1 / 3)
    assert early.mrr > late.mrr


def test_ndcg_rewards_correct_ordering():
    # Both retrieve the same 2 relevant items out of 2, but in different order
    best_order = compute_retrieval_metrics(["a", "b", "x"], {"a", "b"}, k=3)
    worse_order = compute_retrieval_metrics(["x", "a", "b"], {"a", "b"}, k=3)
    assert best_order.ndcg_at_k == 1.0
    assert worse_order.ndcg_at_k < best_order.ndcg_at_k


def test_recall_caps_relevant_count_beyond_k():
    # Only 1 of 2 relevant items fits within k=1
    metrics = compute_retrieval_metrics(["a", "b"], {"a", "b"}, k=1)
    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 1.0


def test_empty_relevant_ids_raises():
    with pytest.raises(ValueError):
        compute_retrieval_metrics(["a"], set(), k=3)


def test_non_positive_k_raises():
    with pytest.raises(ValueError):
        compute_retrieval_metrics(["a"], {"a"}, k=0)


def test_average_metrics_computes_mean_across_cases():
    m1 = compute_retrieval_metrics(["a"], {"a"}, k=1)  # perfect
    m2 = compute_retrieval_metrics(["x"], {"a"}, k=1)  # zero
    averaged = average_metrics([m1, m2])
    assert averaged.recall_at_k == 0.5
    assert averaged.mrr == 0.5


def test_average_metrics_empty_list_raises():
    with pytest.raises(ValueError):
        average_metrics([])
