"""Syncs enterprise entities from PostgreSQL (system of record) into Neo4j
(traversal index over the same entities) — see docs/data-model.md.

Every Neo4j node carries `pg_id` (the PostgreSQL row's UUID as a string),
which is how a graph hit is joined back to full attributes/access-control
metadata. All writes are idempotent MERGEs, so re-running a sync after an
update just refreshes properties/relationships rather than duplicating
nodes.

The neo4j driver is sync, so the actual Cypher execution happens inside
`asyncio.to_thread`; the async parts of this module only do PostgreSQL
reads and shape the data into plain dicts before crossing the thread
boundary (SQLAlchemy ORM objects aren't safe to touch from another thread).
"""

import asyncio
from dataclasses import dataclass, field

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.enterprise import (
    Certification,
    Company,
    Department,
    Employee,
    Industry,
    JobRole,
    Project,
    Skill,
)
from app.services.graph.client import get_neo4j_driver
from app.services.graph.schema import ensure_graph_schema

logger = structlog.get_logger("graph_sync")


@dataclass
class SyncSummary:
    companies: int = 0
    departments: int = 0
    industries: int = 0
    skills: int = 0
    certifications: int = 0
    job_roles: int = 0
    employees: int = 0
    projects: int = 0
    relationships: int = 0
    errors: list[str] = field(default_factory=list)


def _run(driver, query: str, **params) -> None:
    with driver.session() as session:
        session.run(query, **params)


# --- simple label-only entities -------------------------------------------------


def _sync_company_sync(driver, row: dict) -> None:
    _run(
        driver,
        "MERGE (n:Company {pg_id: $pg_id}) SET n.name = $name, n.is_own_org = $is_own_org",
        **row,
    )


def _sync_department_sync(driver, row: dict) -> None:
    _run(driver, "MERGE (n:Department {pg_id: $pg_id}) SET n.name = $name", **row)


def _sync_industry_sync(driver, row: dict) -> None:
    _run(driver, "MERGE (n:Industry {pg_id: $pg_id}) SET n.name = $name", **row)


def _sync_skill_sync(driver, row: dict) -> None:
    _run(
        driver,
        "MERGE (n:Skill {pg_id: $pg_id}) SET n.name = $name, n.category = $category",
        **row,
    )


def _sync_certification_sync(driver, row: dict) -> None:
    _run(
        driver,
        "MERGE (n:Certification {pg_id: $pg_id}) "
        "SET n.name = $name, n.issuing_body = $issuing_body",
        **row,
    )


def _sync_job_role_sync(driver, row: dict) -> None:
    _run(driver, "MERGE (n:JobRole {pg_id: $pg_id}) SET n.title = $title", **row)


# --- employee (node + all its relationships) -------------------------------------


def _sync_employee_sync(driver, employee: dict) -> int:
    relationship_count = 0
    _run(
        driver,
        """
        MERGE (e:Employee {pg_id: $pg_id})
        SET e.full_name = $full_name,
            e.employee_code = $employee_code,
            e.years_experience = $years_experience,
            e.access_level = $access_level
        """,
        pg_id=employee["pg_id"],
        full_name=employee["full_name"],
        employee_code=employee["employee_code"],
        years_experience=employee["years_experience"],
        access_level=employee["access_level"],
    )

    if employee["company_id"]:
        _run(
            driver,
            "MATCH (e:Employee {pg_id: $eid}), (c:Company {pg_id: $cid}) "
            "MERGE (e)-[:WORKS_FOR]->(c)",
            eid=employee["pg_id"],
            cid=employee["company_id"],
        )
        relationship_count += 1

    if employee["department_id"]:
        _run(
            driver,
            "MATCH (e:Employee {pg_id: $eid}), (d:Department {pg_id: $did}) "
            "MERGE (e)-[:WORKS_IN]->(d)",
            eid=employee["pg_id"],
            did=employee["department_id"],
        )
        relationship_count += 1

    for skill in employee["skills"]:
        _run(
            driver,
            "MATCH (e:Employee {pg_id: $eid}), (s:Skill {pg_id: $sid}) "
            "MERGE (e)-[r:HAS_SKILL]->(s) "
            "SET r.years_experience = $years_experience, r.proficiency = $proficiency",
            eid=employee["pg_id"],
            sid=skill["skill_id"],
            years_experience=skill["years_experience"],
            proficiency=skill["proficiency"],
        )
        relationship_count += 1

    for cert in employee["certifications"]:
        _run(
            driver,
            "MATCH (e:Employee {pg_id: $eid}), (c:Certification {pg_id: $cid}) "
            "MERGE (e)-[r:HAS_CERTIFICATION]->(c) "
            "SET r.issued_date = $issued_date, r.expiry_date = $expiry_date",
            eid=employee["pg_id"],
            cid=cert["certification_id"],
            issued_date=cert["issued_date"],
            expiry_date=cert["expiry_date"],
        )
        relationship_count += 1

    for role in employee["roles"]:
        _run(
            driver,
            "MATCH (e:Employee {pg_id: $eid}), (r:JobRole {pg_id: $rid}) "
            "MERGE (e)-[rel:HAS_ROLE]->(r) "
            "SET rel.start_date = $start_date, rel.end_date = $end_date",
            eid=employee["pg_id"],
            rid=role["job_role_id"],
            start_date=role["start_date"],
            end_date=role["end_date"],
        )
        relationship_count += 1

    for proj in employee["projects"]:
        _run(
            driver,
            "MATCH (e:Employee {pg_id: $eid}), (p:Project {pg_id: $pid}) "
            "MERGE (e)-[r:WORKED_ON]->(p) "
            "SET r.role_on_project = $role_on_project",
            eid=employee["pg_id"],
            pid=proj["project_id"],
            role_on_project=proj["role_on_project"],
        )
        relationship_count += 1

    return relationship_count


def _sync_project_sync(driver, project: dict) -> int:
    relationship_count = 0
    _run(
        driver,
        "MERGE (n:Project {pg_id: $pg_id}) SET n.name = $name",
        **{"pg_id": project["pg_id"], "name": project["name"]},
    )

    if project["industry_id"]:
        _run(
            driver,
            "MATCH (p:Project {pg_id: $pid}), (i:Industry {pg_id: $iid}) "
            "MERGE (p)-[:BELONGS_TO]->(i)",
            pid=project["pg_id"],
            iid=project["industry_id"],
        )
        relationship_count += 1

    if project["client_id"]:
        _run(
            driver,
            "MATCH (p:Project {pg_id: $pid}), (c:Company {pg_id: $cid}) "
            "MERGE (p)-[:FOR_CLIENT]->(c)",
            pid=project["pg_id"],
            cid=project["client_id"],
        )
        relationship_count += 1

    return relationship_count


async def sync_all_to_graph(session: AsyncSession) -> SyncSummary:
    summary = SyncSummary()
    driver = get_neo4j_driver()

    companies = (await session.scalars(select(Company))).all()
    departments = (await session.scalars(select(Department))).all()
    industries = (await session.scalars(select(Industry))).all()
    skills = (await session.scalars(select(Skill))).all()
    certifications = (await session.scalars(select(Certification))).all()
    job_roles = (await session.scalars(select(JobRole))).all()

    employees = (
        await session.scalars(
            select(Employee).options(
                selectinload(Employee.skills),
                selectinload(Employee.certifications),
                selectinload(Employee.roles),
                selectinload(Employee.projects),
            )
        )
    ).all()
    projects = (await session.scalars(select(Project))).all()

    def _work():
        ensure_graph_schema()
        for c in companies:
            _sync_company_sync(
                driver, {"pg_id": str(c.id), "name": c.name, "is_own_org": c.is_own_org}
            )
            summary.companies += 1
        for d in departments:
            _sync_department_sync(driver, {"pg_id": str(d.id), "name": d.name})
            summary.departments += 1
        for i in industries:
            _sync_industry_sync(driver, {"pg_id": str(i.id), "name": i.name})
            summary.industries += 1
        for s in skills:
            _sync_skill_sync(driver, {"pg_id": str(s.id), "name": s.name, "category": s.category})
            summary.skills += 1
        for cert in certifications:
            _sync_certification_sync(
                driver,
                {"pg_id": str(cert.id), "name": cert.name, "issuing_body": cert.issuing_body},
            )
            summary.certifications += 1
        for jr in job_roles:
            _sync_job_role_sync(driver, {"pg_id": str(jr.id), "title": jr.title})
            summary.job_roles += 1

        for p in projects:
            summary.relationships += _sync_project_sync(
                driver,
                {
                    "pg_id": str(p.id),
                    "name": p.name,
                    "industry_id": str(p.industry_id) if p.industry_id else None,
                    "client_id": str(p.client_id) if p.client_id else None,
                },
            )
            summary.projects += 1

        for e in employees:
            summary.relationships += _sync_employee_sync(
                driver,
                {
                    "pg_id": str(e.id),
                    "full_name": e.full_name,
                    "employee_code": e.employee_code,
                    "years_experience": e.years_experience,
                    "access_level": e.access_level,
                    "company_id": str(e.company_id) if e.company_id else None,
                    "department_id": str(e.department_id) if e.department_id else None,
                    "skills": [
                        {
                            "skill_id": str(s.skill_id),
                            "years_experience": s.years_experience,
                            "proficiency": s.proficiency,
                        }
                        for s in e.skills
                    ],
                    "certifications": [
                        {
                            "certification_id": str(c.certification_id),
                            "issued_date": c.issued_date,
                            "expiry_date": c.expiry_date,
                        }
                        for c in e.certifications
                    ],
                    "roles": [
                        {
                            "job_role_id": str(r.job_role_id),
                            "start_date": r.start_date,
                            "end_date": r.end_date,
                        }
                        for r in e.roles
                    ],
                    "projects": [
                        {"project_id": str(pr.project_id), "role_on_project": pr.role_on_project}
                        for pr in e.projects
                    ],
                },
            )
            summary.employees += 1

    await asyncio.to_thread(_work)

    logger.info(
        "graph_sync_complete",
        companies=summary.companies,
        departments=summary.departments,
        industries=summary.industries,
        skills=summary.skills,
        certifications=summary.certifications,
        job_roles=summary.job_roles,
        employees=summary.employees,
        projects=summary.projects,
        relationships=summary.relationships,
    )
    return summary
