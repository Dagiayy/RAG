"""Requires postgres + neo4j + qdrant + Ollama all running. Builds an
isolated set of enterprise entities (unique names per run) so this doesn't
depend on scripts/seed_demo_data.py having been run first, and exercises
the full understand_query -> plan_query -> retrieve pipeline end-to-end.
Self-skips (not a fake pass) when Ollama isn't reachable, e.g. in CI.
"""

import uuid

import pytest

from app.models.enterprise import Employee, EmployeeSkill, Industry, Project, Skill
from app.pipelines.query_pipeline import run_query
from app.repositories.db import get_session_factory
from app.services.graph.client import get_neo4j_driver
from app.services.graph.sync import sync_all_to_graph
from tests.conftest import ollama_is_reachable

pytestmark = pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {"employee_ids": [], "project_ids": [], "skill_ids": [], "industry_ids": []}
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for model, key in [
            (Employee, "employee_ids"),
            (Project, "project_ids"),
            (Skill, "skill_ids"),
            (Industry, "industry_ids"),
        ]:
            for entity_id in created[key]:
                obj = await session.get(model, entity_id)
                if obj is not None:
                    await session.delete(obj)
        await session.commit()

    driver = get_neo4j_driver()
    with driver.session() as neo_session:
        for label, key in [
            ("Employee", "employee_ids"),
            ("Project", "project_ids"),
            ("Skill", "skill_ids"),
            ("Industry", "industry_ids"),
        ]:
            for entity_id in created[key]:
                neo_session.run(
                    f"MATCH (n:{label} {{pg_id: $pg_id}}) DETACH DELETE n", pg_id=str(entity_id)
                )


@pytest.mark.asyncio
async def test_pipeline_routes_skill_question_to_graph_and_returns_match(_cleanup):
    skill_name = _unique("Skill")

    factory = get_session_factory()
    async with factory() as session:
        skill = Skill(name=skill_name, category="test")
        session.add(skill)
        await session.flush()
        _cleanup["skill_ids"].append(skill.id)

        employee = Employee(
            employee_code=_unique("EMP"), full_name="Pipeline Test Employee", years_experience=5
        )
        session.add(employee)
        await session.flush()
        _cleanup["employee_ids"].append(employee.id)

        session.add(
            EmployeeSkill(
                employee_id=employee.id, skill_id=skill.id, years_experience=5, proficiency="expert"
            )
        )
        await session.commit()

        await sync_all_to_graph(session)

        # explicitly label it as a skill: a random-suffixed test name (e.g.
        # "Skill-a1b2c3d4") doesn't look like a real skill on its own, and
        # relying on the LLM to infer that from surface form alone was
        # flaky — this phrasing removes that ambiguity.
        result = await run_query(session, f"Which employees have the skill named '{skill_name}'?")

    assert result.intent.intent == "employee_search"
    assert result.plan.use_graph is True
    assert result.plan.use_document_rag is False
    assert any(m.full_name == "Pipeline Test Employee" for m in result.graph_matches)
    assert result.document_evidence == []

    # Phase 9: the generated answer must actually be grounded in the real
    # graph match, not fall back to "insufficient evidence" despite a real
    # match existing (this exact failure mode was caught during manual
    # testing before it shipped — see docs/implementation-status.md).
    assert result.generated.confidence == "evidence_supported"
    assert "Pipeline Test Employee" in result.generated.answer
    assert any(c.employee_name == "Pipeline Test Employee" for c in result.generated.citations)


@pytest.mark.asyncio
async def test_pipeline_document_question_routes_to_hybrid_search():
    # A made-up, uuid-suffixed policy name: unlike a generic topic (e.g.
    # "remote work"), this can never collide with real content that gets
    # permanently seeded into the document corpus (scripts/
    # seed_synthetic_documents.py) — the test's "nothing matches" premise
    # must hold regardless of what else has been ingested, not just at the
    # moment this test was written.
    nonexistent_policy = f"Zyntrevium-{uuid.uuid4().hex[:8]}"

    factory = get_session_factory()
    async with factory() as session:
        result = await run_query(session, f"What is our policy on {nonexistent_policy}?")

    assert result.intent.intent in ("document_qa", "general_qa")
    assert result.plan.use_document_rag is True
    assert result.plan.use_graph is False
    assert result.graph_matches == []

    # No documents about this fictional, uuid-suffixed topic exist (or
    # ever will), so this must correctly report insufficient evidence
    # rather than hallucinate a policy that was never ingested.
    assert result.generated.confidence == "insufficient_evidence"
    assert result.generated.citations == []
