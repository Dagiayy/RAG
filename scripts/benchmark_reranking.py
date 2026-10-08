"""Phase 5 benchmark: does CrossEncoder reranking change/improve ordering
over vector-similarity-only ranking, and what does it cost in latency?

Run: python scripts/benchmark_reranking.py

Self-contained (no Postgres/Qdrant needed) — computes vector-similarity
ranking directly via cosine similarity over the embedding service's output,
then reranks the same candidate set with the CrossEncoder, and prints both
orderings side by side so any reordering is visible and honestly reported
(not asserted as "better" without a labeled relevance judgment, which needs
Phase 11's evaluation dataset — this is a qualitative/latency check).
"""

import time

import numpy as np

from app.services.embeddings.service import EmbeddingService
from app.services.reranking.service import RerankerService

# A deliberately ambiguous case: candidate 2 shares surface words with the
# query ("safety", "procedure") but is about the wrong topic, while
# candidate 3 is genuinely on-topic without repeating the query's exact
# wording. A bi-encoder (vector similarity) can be fooled by surface word
# overlap; a CrossEncoder attends across both texts jointly and should be
# better at telling these apart.
QUERY = "confined space entry safety requirements"
CANDIDATES = [
    ("c1", "Confined space entry requires a permit, gas testing, and a standby attendant."),
    ("c2", "The safety procedure for annual fire drill scheduling is documented separately."),
    ("c3", "Workers entering enclosed tanks or vessels must follow atmospheric monitoring rules."),
    ("c4", "Quarterly financial reporting is unrelated to plant instrumentation work."),
    ("c5", "General workplace safety training must be renewed every two years."),
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    embedder = EmbeddingService()
    reranker = RerankerService()
    print(f"Embedding model: {embedder.model_name}")
    print(f"Reranker model: {reranker.model_name}")
    print(f"Query: {QUERY!r}\n")

    # Warm up both models before timing anything — otherwise the first
    # timed call would include one-time model-load latency (several
    # seconds) mixed in with actual inference time, which would badly
    # mislead the latency numbers below.
    _ = embedder.dimension
    reranker.rerank("warmup", [("w1", "warmup text")])

    start = time.perf_counter()
    query_vector = np.array(embedder.encode_one(QUERY))
    candidate_vectors = {
        chunk_id: np.array(v)
        for chunk_id, v in zip(
            [c[0] for c in CANDIDATES], embedder.encode([c[1] for c in CANDIDATES]), strict=True
        )
    }
    vector_seconds = time.perf_counter() - start

    vector_ranked = sorted(
        CANDIDATES,
        key=lambda c: cosine_similarity(query_vector, candidate_vectors[c[0]]),
        reverse=True,
    )

    start = time.perf_counter()
    rerank_results = reranker.rerank(QUERY, [(c[0], c[1]) for c in CANDIDATES])
    rerank_seconds = time.perf_counter() - start
    rerank_text = {chunk_id: text for chunk_id, text in CANDIDATES}

    print("Vector-similarity ranking (bi-encoder, cosine):")
    for chunk_id, text in vector_ranked:
        score = cosine_similarity(query_vector, candidate_vectors[chunk_id])
        print(f"  {score:.3f}  [{chunk_id}] {text}")

    print("\nCrossEncoder rerank ranking:")
    for chunk_id, score in rerank_results:
        print(f"  {score:.3f}  [{chunk_id}] {rerank_text[chunk_id]}")

    vector_order = [c[0] for c in vector_ranked]
    rerank_order = [chunk_id for chunk_id, _ in rerank_results]
    print(f"\nVector-only order:  {vector_order}")
    print(f"Reranked order:     {rerank_order}")
    print(f"Order changed: {vector_order != rerank_order}")

    n = len(CANDIDATES)
    print(f"\nVector similarity computation: {vector_seconds * 1000:.1f}ms for {n} candidates")
    print(f"CrossEncoder rerank:           {rerank_seconds * 1000:.1f}ms for {n} candidates")


if __name__ == "__main__":
    main()
