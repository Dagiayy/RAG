"""Orchestrates query understanding -> planning -> retrieval -> generation.

`QueryResult.generated` is the grounded, cited answer (Phase 9) — see
app/services/generation/answer_generation.py for the citation-safety
invariant. `graph_matches`/`document_evidence` are kept on the result too
(not just the generated prose) so a caller/UI can show the underlying
evidence for transparency/debugging alongside the answer.
"""

import operator as _operator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.pipelines.query_planner import QueryPlan, describe_graph_query, plan_query
from app.schemas.query_intent import QueryIntent
from app.services.generation.answer_generation import GeneratedAnswer, generate_answer
from app.services.generation.query_understanding import understand_query
from app.services.graph.queries import EmployeeMatch
from app.services.graph.queries import employees_by_certification as graph_by_certification
from app.services.graph.queries import employees_by_client as graph_by_client
from app.services.graph.queries import employees_by_industry as graph_by_industry
from app.services.graph.queries import employees_by_skill as graph_by_skill
from app.services.graph.queries import (
    employees_by_skill_and_industry as graph_by_skill_and_industry,
)
from app.services.graph.queries import project_team as graph_project_team
from app.services.retrieval.hybrid import HybridResult, run_search

_OPERATORS = {
    ">": _operator.gt,
    ">=": _operator.ge,
    "<": _operator.lt,
    "<=": _operator.le,
    "==": _operator.eq,
    "!=": _operator.ne,
}


@dataclass
class QueryResult:
    query: str
    intent: QueryIntent
    plan: QueryPlan
    generated: GeneratedAnswer
    graph_matches: list[EmployeeMatch] = field(default_factory=list)
    document_evidence: list[HybridResult] = field(default_factory=list)


def _apply_filters(matches: list[EmployeeMatch], intent: QueryIntent) -> list[EmployeeMatch]:
    filtered = matches
    for f in intent.filters:
        if f.field == "years_experience":
            op = _OPERATORS[f.operator]
            filtered = [m for m in filtered if op(m.years_experience, f.value)]
    return filtered


async def run_graph_query(
    plan: QueryPlan, intent: QueryIntent, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """Public (not `_`-prefixed) since Phase 10's CV generation pipeline
    reuses this same candidate-selection logic — see
    app/pipelines/cv_pipeline.py. `authorized_levels`: see
    app/services/graph/queries.py's docstrings — None means unrestricted
    (internal callers only); real API callers pass the caller's actual
    authorized AccessLevel values."""
    entities = intent.entities
    match plan.graph_query_type:
        case "by_skill":
            matches = await graph_by_skill(entities["skill"], authorized_levels=authorized_levels)
        case "by_industry":
            matches = await graph_by_industry(
                entities["industry"], authorized_levels=authorized_levels
            )
        case "by_skill_and_industry":
            matches = await graph_by_skill_and_industry(
                entities["skill"], entities["industry"], authorized_levels=authorized_levels
            )
        case "by_certification":
            matches = await graph_by_certification(
                entities["certification"], authorized_levels=authorized_levels
            )
        case "by_client":
            matches = await graph_by_client(entities["client"], authorized_levels=authorized_levels)
        case "project_team":
            matches = await graph_project_team(
                entities["project"], authorized_levels=authorized_levels
            )
        case _:
            matches = []
    return _apply_filters(matches, intent)


async def run_query(
    session: AsyncSession,
    query: str,
    evidence_top_k: int = 8,
    document_filters: dict[str, str | list] | None = None,
    authorized_levels: list[str] | None = None,
) -> QueryResult:
    """`document_filters` (e.g. `{"access_level": [...]}`) is applied to the
    document-RAG path; `authorized_levels` to the graph path — see
    docs/security.md. Callers (the `/query` endpoint) compute both from the
    authenticated user's clearance, never from client input.
    """
    intent = await understand_query(query)
    plan = plan_query(intent)

    graph_matches: list[EmployeeMatch] = []
    document_evidence: list[HybridResult] = []

    if plan.use_graph:
        graph_matches = await run_graph_query(plan, intent, authorized_levels)

    if plan.use_document_rag:
        document_evidence = await run_search(
            session, query, top_k=evidence_top_k, filters=document_filters
        )

    graph_query_description = describe_graph_query(plan, intent) if plan.use_graph else None
    generated = await generate_answer(
        query, graph_matches, document_evidence, graph_query_description
    )

    return QueryResult(
        query=query,
        intent=intent,
        plan=plan,
        generated=generated,
        graph_matches=graph_matches,
        document_evidence=document_evidence,
    )
