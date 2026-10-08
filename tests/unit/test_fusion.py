from app.services.retrieval.fusion import ReciprocalRankFusion


def test_fuse_empty_rankings_returns_empty():
    rrf = ReciprocalRankFusion(k=60)
    assert rrf.fuse({}) == []


def test_fuse_single_retriever_preserves_order():
    rrf = ReciprocalRankFusion(k=60)
    result = rrf.fuse({"vector": ["a", "b", "c"]})
    ids = [item_id for item_id, _ in result]
    assert ids == ["a", "b", "c"]


def test_item_ranked_by_both_retrievers_outranks_single_retriever_hit():
    rrf = ReciprocalRankFusion(k=60)
    result = rrf.fuse(
        {
            "vector": ["a", "b"],
            "bm25": ["a", "c"],
        }
    )
    ids = [item_id for item_id, _ in result]
    assert ids[0] == "a"  # appears at rank 1 in both retrievers


def test_fuse_score_matches_rrf_formula():
    rrf = ReciprocalRankFusion(k=60)
    result = dict(rrf.fuse({"vector": ["a", "b"]}))
    assert result["a"] == 1 / (60 + 1)
    assert result["b"] == 1 / (60 + 2)


def test_fuse_combines_scores_across_retrievers():
    rrf = ReciprocalRankFusion(k=60)
    result = dict(rrf.fuse({"vector": ["a"], "bm25": ["a"]}))
    assert result["a"] == 1 / 61 + 1 / 61


def test_fuse_respects_weights():
    rrf = ReciprocalRankFusion(k=60)
    unweighted = dict(rrf.fuse({"vector": ["a"], "bm25": ["b"]}))
    weighted = dict(
        rrf.fuse({"vector": ["a"], "bm25": ["b"]}, weights={"vector": 2.0, "bm25": 1.0})
    )

    assert unweighted["a"] == unweighted["b"]
    assert weighted["a"] > weighted["b"]


def test_fuse_sorted_descending_by_score():
    rrf = ReciprocalRankFusion(k=60)
    result = rrf.fuse({"vector": ["a", "b", "c", "d"]})
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_negative_k_rejected():
    import pytest

    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=-1)
