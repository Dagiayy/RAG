from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security.access import authorized_access_levels
from app.core.security.audit import log_action
from app.core.security.auth import get_current_user
from app.models.user import User
from app.pipelines.query_pipeline import run_query
from app.repositories.db import get_db_session
from app.schemas.answer import CitationResponse, GeneratedAnswerResponse
from app.schemas.graph import EmployeeMatchResponse
from app.schemas.query import QueryPlanResponse, QueryRequest, QueryResponse
from app.schemas.search import SearchResultItem

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> QueryResponse:
    """Query understanding (LLM -> validated QueryIntent) -> query planning
    (which retriever(s) to run, ADR 0004-compliant fixed Cypher templates
    only) -> retrieval execution -> grounded answer generation with
    citations (spec sections 23-25). `answer.citations` only ever
    references real retrieved evidence — see
    app/services/generation/answer_generation.py. If no evidence was
    found, `answer.confidence` is `"insufficient_evidence"` and the LLM is
    never even called. Underlying evidence (`graph_matches`/
    `document_evidence`) is still returned alongside the answer for
    transparency/debugging.

    Requires a bearer token. Both the document-RAG path (same way `/search`
    is) and the graph path are access-controlled from the authenticated
    user's clearance, computed server-side — never accepted from the
    client (docs/security.md).
    """
    levels = authorized_access_levels(current_user.clearance_level)
    document_filters = {"access_level": levels}
    result = await run_query(
        session, request.query, request.evidence_top_k, document_filters, levels
    )

    await log_action(
        session,
        current_user,
        action="query",
        resource_type="document",
        detail=f"query={request.query[:200]!r}",
    )

    return QueryResponse(
        query=result.query,
        intent=result.intent.intent,
        entities=result.intent.entities,
        plan=QueryPlanResponse(
            use_graph=result.plan.use_graph,
            graph_query_type=result.plan.graph_query_type,
            use_document_rag=result.plan.use_document_rag,
            reason=result.plan.reason,
        ),
        answer=GeneratedAnswerResponse(
            answer=result.generated.answer,
            confidence=result.generated.confidence,
            citations=[
                CitationResponse(
                    marker=c.marker,
                    source_type=c.source_type,
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    source_filename=c.source_filename,
                    page_number=c.page_number,
                    section=c.section,
                    employee_pg_id=c.employee_pg_id,
                    employee_name=c.employee_name,
                )
                for c in result.generated.citations
            ],
        ),
        graph_matches=[
            EmployeeMatchResponse(
                pg_id=m.pg_id,
                full_name=m.full_name,
                employee_code=m.employee_code,
                years_experience=m.years_experience,
                detail=m.detail,
            )
            for m in result.graph_matches
        ],
        document_evidence=[
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
            for r in result.document_evidence
        ],
    )
