"""Hybrid retrieval: vector (Qdrant) + BM25, combined with Reciprocal Rank
Fusion. See docs/retrieval-design.md section "Hybrid fusion".
"""

import asyncio
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.embeddings.service import EmbeddingService, get_embedding_service
from app.services.reranking.service import RerankerService, get_reranker_service
from app.services.retrieval.bm25 import BM25ChunkMetadata, ensure_bm25_index
from app.services.retrieval.fusion import FusionStrategy, ReciprocalRankFusion
from app.services.retrieval.vector import VectorSearchResult
from app.services.retrieval.vector import search as vector_search


@dataclass
class HybridResult:
    chunk_id: str
    document_id: str
    fused_score: float
    vector_score: float | None
    bm25_score: float | None
    text: str
    source_filename: str
    page_number: int | None
    section: str | None
    rerank_score: float | None = None


def _matches_filters(metadata: BM25ChunkMetadata, filters: dict[str, str | list] | None) -> bool:
    if not filters:
        return True
    values = {
        "access_level": metadata.access_level,
        "department": metadata.department,
        "document_status": metadata.document_status,
    }
    for key, expected in filters.items():
        actual = values.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


async def hybrid_search(
    session: AsyncSession,
    query: str,
    top_k: int | None = None,
    filters: dict[str, str | list] | None = None,
    embedding_service: EmbeddingService | None = None,
    fusion_strategy: FusionStrategy | None = None,
) -> list[HybridResult]:
    settings = get_settings()
    top_k = top_k or settings.retrieval_final_evidence_count
    fusion_strategy = fusion_strategy or ReciprocalRankFusion(k=settings.retrieval_rrf_k)
    service = embedding_service or get_embedding_service()

    query_vector = await asyncio.to_thread(service.encode_one, query)

    vector_results, bm25_index = await asyncio.gather(
        asyncio.to_thread(vector_search, query_vector, settings.retrieval_top_k_vector, filters),
        ensure_bm25_index(session),
    )
    bm25_hits = await asyncio.to_thread(bm25_index.search, query, settings.retrieval_top_k_bm25)
    bm25_hits = [
        (chunk_id, score)
        for chunk_id, score in bm25_hits
        if _matches_filters(bm25_index.metadata[chunk_id], filters)
    ]

    rankings = {
        "vector": [r.chunk_id for r in vector_results],
        "bm25": [chunk_id for chunk_id, _ in bm25_hits],
    }
    fused = fusion_strategy.fuse(rankings)

    vector_by_id: dict[str, VectorSearchResult] = {r.chunk_id: r for r in vector_results}
    bm25_score_by_id: dict[str, float] = dict(bm25_hits)

    results: list[HybridResult] = []
    for chunk_id, fused_score in fused[:top_k]:
        vec = vector_by_id.get(chunk_id)
        bm25_meta = bm25_index.metadata.get(chunk_id)

        if vec is not None:
            document_id = vec.document_id
            text = vec.payload.get("text", "")
            source_filename = vec.payload.get("source_filename", "")
            page_number = vec.payload.get("page_number")
            section = vec.payload.get("section")
        elif bm25_meta is not None:
            document_id = bm25_meta.document_id
            text = bm25_meta.text
            source_filename = bm25_meta.source_filename
            page_number = bm25_meta.page_number
            section = bm25_meta.section
        else:
            continue  # shouldn't happen: chunk_id came from one of the two rankings

        results.append(
            HybridResult(
                chunk_id=chunk_id,
                document_id=document_id,
                fused_score=fused_score,
                vector_score=vec.score if vec else None,
                bm25_score=bm25_score_by_id.get(chunk_id),
                text=text,
                source_filename=source_filename,
                page_number=page_number,
                section=section,
            )
        )

    return results


async def _fetch_candidates(
    session: AsyncSession,
    query: str,
    mode: str,
    candidate_k: int,
    filters: dict[str, str | list] | None,
    embedding_service: EmbeddingService,
    score_threshold: float | None,
) -> list[HybridResult]:
    if mode == "hybrid":
        return await hybrid_search(session, query, candidate_k, filters, embedding_service)

    if mode == "vector":
        query_vector = await asyncio.to_thread(embedding_service.encode_one, query)
        vector_results = await asyncio.to_thread(
            vector_search, query_vector, candidate_k, filters, score_threshold
        )
        return [
            HybridResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                fused_score=r.score,
                vector_score=r.score,
                bm25_score=None,
                text=r.payload.get("text", ""),
                source_filename=r.payload.get("source_filename", ""),
                page_number=r.payload.get("page_number"),
                section=r.payload.get("section"),
            )
            for r in vector_results
        ]

    if mode == "bm25":
        bm25_index = await ensure_bm25_index(session)
        hits = await asyncio.to_thread(bm25_index.search, query, candidate_k)
        hits = [
            (chunk_id, score)
            for chunk_id, score in hits
            if _matches_filters(bm25_index.metadata[chunk_id], filters)
        ]
        return [
            HybridResult(
                chunk_id=chunk_id,
                document_id=bm25_index.metadata[chunk_id].document_id,
                fused_score=score,
                vector_score=None,
                bm25_score=score,
                text=bm25_index.metadata[chunk_id].text,
                source_filename=bm25_index.metadata[chunk_id].source_filename,
                page_number=bm25_index.metadata[chunk_id].page_number,
                section=bm25_index.metadata[chunk_id].section,
            )
            for chunk_id, score in hits
        ]

    raise ValueError(f"Unknown search mode: {mode}")


async def run_search(
    session: AsyncSession,
    query: str,
    mode: str = "hybrid",
    top_k: int | None = None,
    filters: dict[str, str | list] | None = None,
    embedding_service: EmbeddingService | None = None,
    score_threshold: float | None = None,
    rerank: bool = True,
    reranker_service: RerankerService | None = None,
    include_historical: bool = False,
) -> list[HybridResult]:
    """Dispatches candidate generation to hybrid/vector-only/bm25-only
    (always returning the same HybridResult shape — score fields for the
    retriever(s) not used are left None — so callers can compare modes
    without different response shapes), then optionally reranks with a
    CrossEncoder (spec section 12: retrieve a larger candidate pool, rerank,
    truncate to the final evidence count). See docs/retrieval-design.md.

    `score_threshold` only applies to mode="vector" (raw cosine similarity
    has a well-defined scale). RRF-fused and BM25 raw scores don't have a
    comparable fixed scale, so it's ignored for "hybrid"/"bm25" — filter
    those client-side on the returned scores if needed.

    `include_historical`: by default, retrieval prefers the currently
    effective version of a document (spec section 20) — superseded
    versions are excluded unless this is True or `filters` already
    specifies `document_status` explicitly (e.g. to query a specific
    historical status).
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_final_evidence_count
    service = embedding_service or get_embedding_service()

    merged_filters = dict(filters) if filters else {}
    if not include_historical and "document_status" not in merged_filters:
        merged_filters["document_status"] = "active"

    candidate_k = max(top_k, settings.retrieval_rerank_candidates) if rerank else top_k
    candidates = await _fetch_candidates(
        session, query, mode, candidate_k, merged_filters, service, score_threshold
    )

    if not rerank or not candidates:
        return candidates[:top_k]

    reranker = reranker_service or get_reranker_service()
    pairs = [(c.chunk_id, c.text) for c in candidates]
    reranked = await asyncio.to_thread(reranker.rerank, query, pairs, top_k)

    by_id = {c.chunk_id: c for c in candidates}
    results = []
    for chunk_id, score in reranked:
        candidate = by_id[chunk_id]
        candidate.rerank_score = score
        results.append(candidate)
    return results
