"""Requires postgres + neo4j running. Builds a small isolated graph (unique
names per test run) and exercises each Cypher query template in
app/services/graph/queries.py.
"""

import uuid

import pytest

from app.models.enterprise import (
    Company,
    EmployeeProject,
    EmployeeSkill,
    Industry,
    Project,
    Skill,
)
from app.models.enterprise import Employee as EmployeeModel
from app.repositories.db import get_session_factory
from app.services.graph.client import get_neo4j_driver
from app.services.graph.queries import (
    employees_by_client,
    employees_by_industry,
    employees_by_skill,
    employees_by_skill_and_industry,
    project_team,
)
from app.services.graph.sync import sync_all_to_graph


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {
        "employee_ids": [],
        "project_ids": [],
        "company_ids": [],
        "skill_ids": [],
        "industry_ids": [],
    }
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for model, key in [
            (EmployeeModel, "employee_ids"),
            (Project, "project_ids"),
            (Company, "company_ids"),
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
            ("Company", "company_ids"),
            ("Skill", "skill_ids"),
            ("Industry", "industry_ids"),
        ]:
            for entity_id in created[key]:
                neo_session.run(
                    f"MATCH (n:{label} {{pg_id: $pg_id}}) DETACH DELETE n", pg_id=str(entity_id)
                )


@pytest.mark.asyncio
async def test_graph_query_shapes(_cleanup):
    skill_name = _unique("Skill")
    industry_name = _unique("Industry")
    client_name = _unique("Client")
    project_name = _unique("Project")

    factory = get_session_factory()
    async with factory() as session:
        skill = Skill(name=skill_name, category="test")
        industry = Industry(name=industry_name)
        client = Company(name=client_name, is_own_org=False)
        session.add_all([skill, industry, client])
        await session.flush()
        _cleanup["skill_ids"].append(skill.id)
        _cleanup["industry_ids"].append(industry.id)
        _cleanup["company_ids"].append(client.id)

        project = Project(name=project_name, industry_id=industry.id, client_id=client.id)
        session.add(project)
        await session.flush()
        _cleanup["project_ids"].append(project.id)

        matching_employee = EmployeeModel(
            employee_code=_unique("EMP"), full_name="Matching Employee", years_experience=5
        )
        non_matching_employee = EmployeeModel(
            employee_code=_unique("EMP"), full_name="Non Matching Employee", years_experience=2
        )
        session.add_all([matching_employee, non_matching_employee])
        await session.flush()
        _cleanup["employee_ids"].extend([matching_employee.id, non_matching_employee.id])

        session.add(
            EmployeeSkill(
                employee_id=matching_employee.id,
                skill_id=skill.id,
                years_experience=5,
                proficiency="expert",
            )
        )
        session.add(
            EmployeeProject(
                employee_id=matching_employee.id, project_id=project.id, role_on_project="Lead"
            )
        )
        await session.commit()

        await sync_all_to_graph(session)

    by_skill = await employees_by_skill(skill_name)
    assert {e.full_name for e in by_skill} == {"Matching Employee"}

    by_industry = await employees_by_industry(industry_name)
    assert {e.full_name for e in by_industry} == {"Matching Employee"}

    by_skill_and_industry = await employees_by_skill_and_industry(skill_name, industry_name)
    assert {e.full_name for e in by_skill_and_industry} == {"Matching Employee"}

    by_client = await employees_by_client(client_name)
    assert {e.full_name for e in by_client} == {"Matching Employee"}

    team = await project_team(project_name)
    assert len(team) == 1
    assert team[0].full_name == "Matching Employee"
    assert team[0].detail == "Lead"


@pytest.mark.asyncio
async def test_employees_by_skill_respects_min_years(_cleanup):
    skill_name = _unique("Skill")

    factory = get_session_factory()
    async with factory() as session:
        skill = Skill(name=skill_name, category="test")
        session.add(skill)
        await session.flush()
        _cleanup["skill_ids"].append(skill.id)

        junior = EmployeeModel(
            employee_code=_unique("EMP"), full_name="Junior Employee", years_experience=1
        )
        senior = EmployeeModel(
            employee_code=_unique("EMP"), full_name="Senior Employee", years_experience=8
        )
        session.add_all([junior, senior])
        await session.flush()
        _cleanup["employee_ids"].extend([junior.id, senior.id])

        session.add(
            EmployeeSkill(
                employee_id=junior.id, skill_id=skill.id, years_experience=1, proficiency="beginner"
            )
        )
        session.add(
            EmployeeSkill(
                employee_id=senior.id, skill_id=skill.id, years_experience=8, proficiency="expert"
            )
        )
        await session.commit()

        await sync_all_to_graph(session)

    results = await employees_by_skill(skill_name, min_years=5)
    assert {e.full_name for e in results} == {"Senior Employee"}


@pytest.mark.asyncio
async def test_query_for_nonexistent_skill_returns_empty():
    results = await employees_by_skill(_unique("NoSuchSkill"))
    assert results == []
