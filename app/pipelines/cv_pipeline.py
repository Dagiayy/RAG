"""CV generation candidate selection (spec section 26):

natural-language query -> understand_query -> plan_query -> graph search
(reusing Phase 7's candidate-selection logic) -> full EmployeeProfile per
candidate.

Deliberately does NOT call the LLM at any point — candidate selection is
the same deterministic Cypher-template routing used by `/query`, and
profile assembly is a direct graph read. See
app/services/documents/cv_generator.py for why rendering skips the LLM
too. This keeps "must not invent employee experience" (spec section 26)
an architectural property, not a hope.
"""

import asyncio

from app.pipelines.query_pipeline import run_graph_query
from app.pipelines.query_planner import plan_query
from app.services.generation.query_understanding import understand_query
from app.services.graph.queries import EmployeeProfile, get_employee_profile


async def select_cv_candidates(
    query: str, authorized_levels: list[str] | None = None
) -> list[EmployeeProfile]:
    """Returns full profiles for every employee matching the natural-
    language query's graph-queryable criteria. Returns an empty list (not
    an error) if the query doesn't route to a graph query at all, or if it
    does but nothing matches — spec section 26's pipeline is
    understanding -> candidate selection -> ... -> validation; an empty
    candidate list is a valid, honest outcome, not a failure.
    `authorized_levels`: see app/services/graph/queries.py's docstrings —
    a CV can't be generated for an employee the caller isn't authorized to
    see, so this is threaded through both the candidate search and the
    profile fetch."""
    intent = await understand_query(query)
    plan = plan_query(intent)

    if not plan.use_graph:
        return []

    matches = await run_graph_query(plan, intent, authorized_levels)
    profiles = await asyncio.gather(
        *(get_employee_profile(m.pg_id, authorized_levels) for m in matches)
    )
    return [p for p in profiles if p is not None]
