# ADR 0003: Reciprocal Rank Fusion for hybrid retrieval

## Status
Accepted

## Context
Vector similarity (cosine, roughly 0-1) and BM25 scores (unbounded,
corpus-dependent) are not on comparable scales. Naive min-max normalization
per query is sensitive to outliers and score distribution shifts between
queries.

## Decision
Combine ranked lists with Reciprocal Rank Fusion:
`score(d) = sum_r 1/(k + rank_r(d))`, `k=60` default, configurable per
retriever weight. Implemented behind a `FusionStrategy` interface
(`app/services/retrieval/fusion.py`) so alternative strategies (weighted-sum
after normalization, learned fusion) can be swapped in and A/B evaluated
without touching callers.

## Consequences
RRF needs no score calibration and is robust to scale mismatches, at the
cost of discarding score magnitude information. Acceptable since reranking
(CrossEncoder) is the stage that actually determines final ordering quality.
