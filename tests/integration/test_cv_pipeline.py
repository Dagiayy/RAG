"""Requires postgres + neo4j + Ollama running. Self-skips when Ollama
isn't reachable (see tests/conftest.py).
"""

import uuid

import pytest

from app.models.enterprise import Employee, EmployeeSkill, Skill
from app.pipelines.cv_pipeline import select_cv_candidates
from app.repositories.db import get_session_factory
from app.services.graph.client import get_neo4j_driver
from app.services.graph.sync import sync_all_to_graph
from tests.conftest import ollama_is_reachable

pytestmark = pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {"employee_ids": [], "skill_ids": []}
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for model, key in [(Employee, "employee_ids"), (Skill, "skill_ids")]:
            for entity_id in created[key]:
                obj = await session.get(model, entity_id)
                if obj is not None:
                    await session.delete(obj)
        await session.commit()

    driver = get_neo4j_driver()
    with driver.session() as neo_session:
        for label, key in [("Employee", "employee_ids"), ("Skill", "skill_ids")]:
            for entity_id in created[key]:
                neo_session.run(
                    f"MATCH (n:{label} {{pg_id: $pg_id}}) DETACH DELETE n", pg_id=str(entity_id)
                )


@pytest.mark.asyncio
async def test_select_cv_candidates_returns_full_profiles_for_matches(_cleanup):
    skill_name = _unique("Skill")

    factory = get_session_factory()
    async with factory() as session:
        skill = Skill(name=skill_name, category="test")
        session.add(skill)
        await session.flush()
        _cleanup["skill_ids"].append(skill.id)

        employee = Employee(
            employee_code=_unique("EMP"), full_name="CV Candidate Employee", years_experience=5
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

    profiles = await select_cv_candidates(
        f"Generate CVs for employees with the skill '{skill_name}'"
    )

    assert len(profiles) == 1
    assert profiles[0].full_name == "CV Candidate Employee"
    assert any(s.name == skill_name for s in profiles[0].skills)


@pytest.mark.asyncio
async def test_select_cv_candidates_returns_empty_for_document_question():
    profiles = await select_cv_candidates("What is our remote work policy?")
    assert profiles == []
