"""Requires postgres + neo4j running. Builds an isolated employee with a
full set of relationships to verify get_employee_profile() assembles them
correctly, including edge cases (no relationships, nonexistent pg_id).
"""

import uuid

import pytest

from app.models.enterprise import (
    Certification,
    Company,
    Department,
    Employee,
    EmployeeCertification,
    EmployeeProject,
    EmployeeRole,
    EmployeeSkill,
    Industry,
    JobRole,
    Project,
    Skill,
)
from app.repositories.db import get_session_factory
from app.services.graph.client import get_neo4j_driver
from app.services.graph.queries import get_employee_profile
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
        "certification_ids": [],
        "job_role_ids": [],
    }
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for model, key in [
            (Employee, "employee_ids"),
            (Project, "project_ids"),
            (Company, "company_ids"),
            (Department, "department_ids"),
            (Industry, "industry_ids"),
            (Skill, "skill_ids"),
            (Certification, "certification_ids"),
            (JobRole, "job_role_ids"),
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
            ("Department", "department_ids"),
            ("Industry", "industry_ids"),
            ("Skill", "skill_ids"),
            ("Certification", "certification_ids"),
            ("JobRole", "job_role_ids"),
        ]:
            for entity_id in created[key]:
                neo_session.run(
                    f"MATCH (n:{label} {{pg_id: $pg_id}}) DETACH DELETE n", pg_id=str(entity_id)
                )


@pytest.mark.asyncio
async def test_full_profile_includes_all_relationships(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        company = Company(name=_unique("Company"), is_own_org=True)
        department = Department(name=_unique("Dept"))
        skill = Skill(name=_unique("Skill"), category="test")
        certification = Certification(name=_unique("Cert"), issuing_body="Test Body")
        job_role = JobRole(title=_unique("Role"))
        industry = Industry(name=_unique("Industry"))
        session.add_all([company, department, skill, certification, job_role, industry])
        await session.flush()
        _cleanup["company_ids"].append(company.id)
        _cleanup["department_ids"].append(department.id)
        _cleanup["skill_ids"].append(skill.id)
        _cleanup["certification_ids"].append(certification.id)
        _cleanup["job_role_ids"].append(job_role.id)
        _cleanup["industry_ids"].append(industry.id)

        project = Project(name=_unique("Project"), industry_id=industry.id, client_id=company.id)
        session.add(project)
        await session.flush()
        _cleanup["project_ids"].append(project.id)

        employee = Employee(
            employee_code=_unique("EMP"),
            full_name="Full Profile Employee",
            years_experience=8,
            department_id=department.id,
            company_id=company.id,
        )
        session.add(employee)
        await session.flush()
        _cleanup["employee_ids"].append(employee.id)

        session.add(
            EmployeeSkill(
                employee_id=employee.id, skill_id=skill.id, years_experience=5, proficiency="expert"
            )
        )
        session.add(
            EmployeeCertification(
                employee_id=employee.id,
                certification_id=certification.id,
                issued_date=None,
                expiry_date=None,
            )
        )
        session.add(EmployeeRole(employee_id=employee.id, job_role_id=job_role.id))
        session.add(
            EmployeeProject(employee_id=employee.id, project_id=project.id, role_on_project="Lead")
        )
        await session.commit()

        await sync_all_to_graph(session)

        profile = await get_employee_profile(str(employee.id))

    assert profile is not None
    assert profile.full_name == "Full Profile Employee"
    assert profile.years_experience == 8
    assert profile.company_name == company.name
    assert profile.department_name == department.name
    assert len(profile.skills) == 1
    assert profile.skills[0].name == skill.name
    assert profile.skills[0].proficiency == "expert"
    assert len(profile.certifications) == 1
    assert profile.certifications[0].name == certification.name
    assert len(profile.roles) == 1
    assert profile.roles[0].title == job_role.title
    assert len(profile.projects) == 1
    assert profile.projects[0].name == project.name
    assert profile.projects[0].role_on_project == "Lead"
    assert profile.projects[0].industry == industry.name
    assert profile.projects[0].client == company.name


@pytest.mark.asyncio
async def test_profile_for_employee_with_no_relationships_has_empty_lists(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        employee = Employee(
            employee_code=_unique("EMP"), full_name="Bare Employee", years_experience=1
        )
        session.add(employee)
        await session.flush()
        _cleanup["employee_ids"].append(employee.id)
        await session.commit()

        await sync_all_to_graph(session)

        profile = await get_employee_profile(str(employee.id))

    assert profile is not None
    assert profile.full_name == "Bare Employee"
    assert profile.skills == []
    assert profile.certifications == []
    assert profile.roles == []
    assert profile.projects == []
    assert profile.company_name is None
    assert profile.department_name is None


@pytest.mark.asyncio
async def test_profile_for_nonexistent_pg_id_returns_none():
    profile = await get_employee_profile(str(uuid.uuid4()))
    assert profile is None
