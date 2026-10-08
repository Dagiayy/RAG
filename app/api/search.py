from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security.access import authorized_access_levels
from app.core.security.audit import log_action
from app.core.security.auth import get_current_user
from app.models.user import User
from app.repositories.db import get_db_session
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.services.retrieval.hybrid import run_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> SearchResponse:
    """Hybrid (vector + BM25, RRF-fused) search by default, reranked with a
    CrossEncoder over a wider candidate pool (spec section 12). Pass
    `mode: "vector"` or `mode: "bm25"` to compare against a single
    retriever, and `rerank: false` to see pre-rerank ordering.

    Requires a bearer token (`Authorization: Bearer <api_key>` — see
    `scripts/seed_users.py`). The access-level filter is computed from the
    authenticated user's clearance and applied inside the retrieval query
    itself, before any content is assembled — never accepted from the
    client and never enforced by post-hoc filtering (docs/security.md).
    """
    filters: dict[str, str | list] = {
        "access_level": authorized_access_levels(current_user.clearance_level)
    }
    if request.department:
        filters["department"] = request.department

    results = await run_search(
        session,
        request.query,
        mode=request.mode,
        top_k=request.top_k,
        filters=filters,
        score_threshold=request.score_threshold,
        rerank=request.rerank,
        include_historical=request.include_historical,
    )

    await log_action(
        session,
        current_user,
        action="search",
        resource_type="document",
        detail=f"query={request.query[:200]!r}",
    )

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        reranked=request.rerank,
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                score=r.rerank_score if r.rerank_score is not None else r.fused_score,
                vector_score=r.vector_score,
                bm25_score=r.bm25_score,
                rerank_score=r.rerank_score,
                text=r.text,
                source_filename=r.source_filename,
                page_number=r.page_number,
                section=r.section,
            )
            for r in results
        ],
    )
