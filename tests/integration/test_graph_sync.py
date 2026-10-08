"""Requires postgres + neo4j running (`docker compose up -d postgres neo4j`)
and migrations applied. Seeds a small, isolated set of enterprise entities
(distinct names from scripts/seed_demo_data.py's fixtures, to avoid
collisions if that script has already been run against the same DB) and
verifies both the PostgreSQL rows and the resulting Neo4j graph.
"""

import uuid

import pytest

from app.models.enterprise import (
    Company,
    Department,
    Employee,
    EmployeeProject,
    EmployeeSkill,
    Industry,
    Project,
    Skill,
)
from app.repositories.db import get_session_factory
from app.services.graph.client import get_neo4j_driver
from app.services.graph.sync import sync_all_to_graph


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {
        "employee_ids": [],
        "project_ids": [],
        "company_ids": [],
        "department_ids": [],
        "industry_ids": [],
        "skill_ids": [],
    }
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for model, ids_key in [
            (Employee, "employee_ids"),
            (Project, "project_ids"),
            (Company, "company_ids"),
            (Department, "department_ids"),
            (Industry, "industry_ids"),
            (Skill, "skill_ids"),
        ]:
            for entity_id in created[ids_key]:
                obj = await session.get(model, entity_id)
                if obj is not None:
                    await session.delete(obj)
        await session.commit()

    driver = get_neo4j_driver()
    with driver.session() as neo_session:
        for label, ids_key in [
            ("Employee", "employee_ids"),
            ("Project", "project_ids"),
            ("Company", "company_ids"),
            ("Department", "department_ids"),
            ("Industry", "industry_ids"),
            ("Skill", "skill_ids"),
        ]:
            for entity_id in created[ids_key]:
                neo_session.run(
                    f"MATCH (n:{label} {{pg_id: $pg_id}}) DETACH DELETE n", pg_id=str(entity_id)
                )


@pytest.mark.asyncio
async def test_sync_creates_nodes_and_relationships(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        company = Company(name=_unique("TestCo"), is_own_org=True)
        department = Department(name=_unique("TestDept"))
        industry = Industry(name=_unique("TestIndustry"))
        skill = Skill(name=_unique("TestSkill"), category="test")
        session.add_all([company, department, industry, skill])
        await session.flush()
        _cleanup["company_ids"].append(company.id)
        _cleanup["department_ids"].append(department.id)
        _cleanup["industry_ids"].append(industry.id)
        _cleanup["skill_ids"].append(skill.id)

        project = Project(
            name=_unique("TestProject"), company_id=company.id, industry_id=industry.id
        )
        session.add(project)
        await session.flush()
        _cleanup["project_ids"].append(project.id)

        employee = Employee(
            employee_code=_unique("EMP"),
            full_name="Test Employee",
            years_experience=3,
            department_id=department.id,
            company_id=company.id,
        )
        session.add(employee)
        await session.flush()
        _cleanup["employee_ids"].append(employee.id)

        session.add(
            EmployeeSkill(
                employee_id=employee.id, skill_id=skill.id, years_experience=3, proficiency="expert"
            )
        )
        session.add(
            EmployeeProject(
                employee_id=employee.id, project_id=project.id, role_on_project="Tester"
            )
        )
        await session.commit()

        summary = await sync_all_to_graph(session)
        assert summary.employees >= 1
        assert summary.relationships >= 1

    driver = get_neo4j_driver()
    with driver.session() as neo_session:
        record = neo_session.run(
            "MATCH (e:Employee {pg_id: $pg_id})-[:HAS_SKILL]->(s:Skill) "
            "RETURN s.name AS skill_name",
            pg_id=str(employee.id),
        ).single()
        assert record is not None
        assert record["skill_name"] == skill.name

        record = neo_session.run(
            "MATCH (e:Employee {pg_id: $pg_id})-[:WORKED_ON]->(p:Project)"
            "-[:BELONGS_TO]->(i:Industry) RETURN i.name AS industry_name",
            pg_id=str(employee.id),
        ).single()
        assert record is not None
        assert record["industry_name"] == industry.name


@pytest.mark.asyncio
async def test_sync_is_idempotent(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        company = Company(name=_unique("TestCo"), is_own_org=True)
        session.add(company)
        await session.flush()
        _cleanup["company_ids"].append(company.id)
        await session.commit()

        await sync_all_to_graph(session)
        await sync_all_to_graph(session)

    driver = get_neo4j_driver()
    with driver.session() as neo_session:
        count = neo_session.run(
            "MATCH (c:Company {pg_id: $pg_id}) RETURN count(c) AS count", pg_id=str(company.id)
        ).single()["count"]
        assert count == 1
