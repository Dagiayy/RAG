"""Maps a validated QueryIntent to which retriever(s) to run (spec section
16). Deliberately avoids unnecessary retrieval: a pure entity-graph
question (skill/industry/certification/client/project) never triggers
document RAG, and a document question never triggers a graph query.

Routing is entity-presence-based, not just intent-based, because the same
`employee_search` intent needs a different Cypher template depending on
which entities were actually extracted (skill alone vs. skill+industry —
see app/services/graph/queries.py's templates). Multi-hop (skill AND
industry) is only used when both are present; ADR 0004 still applies — no
free-form Cypher generation, just picking among fixed templates.
"""

from dataclasses import dataclass
from typing import Literal

from app.schemas.query_intent import QueryIntent

GraphQueryType = Literal[
    "by_skill",
    "by_industry",
    "by_skill_and_industry",
    "by_certification",
    "by_client",
    "project_team",
]


@dataclass
class QueryPlan:
    use_graph: bool
    graph_query_type: GraphQueryType | None
    use_document_rag: bool
    reason: str


def plan_query(intent: QueryIntent) -> QueryPlan:
    entities = intent.entities

    if intent.intent in ("employee_search", "comparison", "cv_generation"):
        if "skill" in entities and "industry" in entities:
            return QueryPlan(
                True,
                "by_skill_and_industry",
                False,
                "skill+industry entities -> multi-hop graph query",
            )
        if "skill" in entities:
            return QueryPlan(True, "by_skill", False, "skill entity -> graph skill query")
        if "industry" in entities:
            return QueryPlan(True, "by_industry", False, "industry entity -> graph industry query")
        if "certification" in entities:
            return QueryPlan(
                True, "by_certification", False, "certification entity -> graph certification query"
            )
        if "client" in entities:
            return QueryPlan(True, "by_client", False, "client entity -> graph client query")
        return QueryPlan(
            False,
            None,
            True,
            f"{intent.intent} intent but no graph-queryable entities -> document RAG",
        )

    if intent.intent == "project_lookup" and "project" in entities:
        return QueryPlan(True, "project_team", False, "project entity -> graph project-team query")

    if intent.intent == "document_qa":
        return QueryPlan(False, None, True, "document_qa intent -> hybrid document retrieval")

    return QueryPlan(
        False, None, True, "general_qa or unrouted intent -> hybrid document retrieval"
    )


def describe_graph_query(plan: QueryPlan, intent: QueryIntent) -> str | None:
    """Human-readable description of what a graph query searched for, e.g.
    "employees with skill 'GIS Mapping'". Embedded into generated evidence
    text (app/services/generation/context_builder.py) so the answer-
    generation LLM can ground a match back to the question — without it,
    an employee record's raw attributes never literally restate what was
    searched for, and grounded generation incorrectly treats real matches
    as insufficient evidence (see docs/implementation-status.md)."""
    entities = intent.entities
    match plan.graph_query_type:
        case "by_skill":
            return f"employees with skill '{entities.get('skill')}'"
        case "by_industry":
            return f"employees who worked on projects in industry '{entities.get('industry')}'"
        case "by_skill_and_industry":
            return (
                f"employees with skill '{entities.get('skill')}' who worked on projects "
                f"in industry '{entities.get('industry')}'"
            )
        case "by_certification":
            return f"employees holding certification '{entities.get('certification')}'"
        case "by_client":
            return f"employees who worked on projects for client '{entities.get('client')}'"
        case "project_team":
            return f"employees who worked on project '{entities.get('project')}'"
        case _:
            return None
