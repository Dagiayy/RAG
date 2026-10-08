"""Parameterized Cypher templates for graph retrieval (see ADR 0004: no
free-form LLM-generated Cypher against the live graph). Each function here
covers one of the documented question shapes from docs/architecture.md
section 13; new question shapes get new templates, not query-generation
capability handed to the LLM. All queries run through a read-only session.

Every query runs via `asyncio.to_thread` since the neo4j driver is sync.
"""

import asyncio
from dataclasses import dataclass

from app.services.graph.client import get_neo4j_driver


@dataclass
class EmployeeMatch:
    pg_id: str
    full_name: str
    employee_code: str
    years_experience: int
    detail: str | None = None  # e.g. matched skill's years_experience/proficiency


def _run_read(query: str, **params) -> list[dict]:
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


async def employees_by_skill(
    skill_name: str, min_years: int | None = None, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """'Which employees have <skill> experience?' (optionally filtered by
    years of experience with that specific skill). `authorized_levels`:
    the caller's authorized AccessLevel values (app/core/security/access.py)
    — None means unrestricted, used by internal callers (tests, scripts)
    that don't go through an authenticated API path; every real endpoint
    passes the caller's actual authorized levels (docs/security.md)."""
    query = """
        MATCH (e:Employee)-[r:HAS_SKILL]->(s:Skill)
        WHERE toLower(s.name) = toLower($skill_name)
          AND ($min_years IS NULL OR r.years_experience >= $min_years)
          AND ($authorized_levels IS NULL OR e.access_level IN $authorized_levels)
        RETURN e.pg_id AS pg_id, e.full_name AS full_name, e.employee_code AS employee_code,
               e.years_experience AS years_experience, r.proficiency AS proficiency,
               r.years_experience AS skill_years
        ORDER BY r.years_experience DESC
    """
    rows = await asyncio.to_thread(
        _run_read,
        query,
        skill_name=skill_name,
        min_years=min_years,
        authorized_levels=authorized_levels,
    )
    return [
        EmployeeMatch(
            pg_id=row["pg_id"],
            full_name=row["full_name"],
            employee_code=row["employee_code"],
            years_experience=row["years_experience"],
            detail=f"{row['skill_years']}y, {row['proficiency']}" if row["proficiency"] else None,
        )
        for row in rows
    ]


async def employees_by_industry(
    industry_name: str, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """'Which employees have worked on <industry> projects?' — traverses
    Employee -> WORKED_ON -> Project -> BELONGS_TO -> Industry."""
    query = """
        MATCH (e:Employee)-[:WORKED_ON]->(p:Project)-[:BELONGS_TO]->(i:Industry)
        WHERE toLower(i.name) = toLower($industry_name)
          AND ($authorized_levels IS NULL OR e.access_level IN $authorized_levels)
        RETURN DISTINCT e.pg_id AS pg_id, e.full_name AS full_name,
               e.employee_code AS employee_code, e.years_experience AS years_experience,
               collect(DISTINCT p.name) AS project_names
    """
    rows = await asyncio.to_thread(
        _run_read, query, industry_name=industry_name, authorized_levels=authorized_levels
    )
    return [
        EmployeeMatch(
            pg_id=row["pg_id"],
            full_name=row["full_name"],
            employee_code=row["employee_code"],
            years_experience=row["years_experience"],
            detail=", ".join(row["project_names"]),
        )
        for row in rows
    ]


async def employees_by_skill_and_industry(
    skill_name: str, industry_name: str, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """Multi-hop: employees with a given skill AND who worked on a project
    in a given industry (spec section 14/17 example)."""
    query = """
        MATCH (e:Employee)-[:HAS_SKILL]->(s:Skill)
        WHERE toLower(s.name) = toLower($skill_name)
          AND ($authorized_levels IS NULL OR e.access_level IN $authorized_levels)
        MATCH (e)-[:WORKED_ON]->(p:Project)-[:BELONGS_TO]->(i:Industry)
        WHERE toLower(i.name) = toLower($industry_name)
        RETURN DISTINCT e.pg_id AS pg_id, e.full_name AS full_name,
               e.employee_code AS employee_code, e.years_experience AS years_experience,
               collect(DISTINCT p.name) AS project_names
    """
    rows = await asyncio.to_thread(
        _run_read,
        query,
        skill_name=skill_name,
        industry_name=industry_name,
        authorized_levels=authorized_levels,
    )
    return [
        EmployeeMatch(
            pg_id=row["pg_id"],
            full_name=row["full_name"],
            employee_code=row["employee_code"],
            years_experience=row["years_experience"],
            detail=", ".join(row["project_names"]),
        )
        for row in rows
    ]


async def employees_by_certification(
    certification_name: str, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """'What certifications does Employee A have?' in reverse — 'which
    employees hold <certification>?'"""
    query = """
        MATCH (e:Employee)-[r:HAS_CERTIFICATION]->(c:Certification)
        WHERE toLower(c.name) = toLower($certification_name)
          AND ($authorized_levels IS NULL OR e.access_level IN $authorized_levels)
        RETURN e.pg_id AS pg_id, e.full_name AS full_name, e.employee_code AS employee_code,
               e.years_experience AS years_experience, r.expiry_date AS expiry_date
    """
    rows = await asyncio.to_thread(
        _run_read,
        query,
        certification_name=certification_name,
        authorized_levels=authorized_levels,
    )
    return [
        EmployeeMatch(
            pg_id=row["pg_id"],
            full_name=row["full_name"],
            employee_code=row["employee_code"],
            years_experience=row["years_experience"],
            detail=f"expires {row['expiry_date']}" if row["expiry_date"] else None,
        )
        for row in rows
    ]


async def employees_by_client(
    client_name: str, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """'Which employees worked on projects for <client>?' — traverses
    Employee -> WORKED_ON -> Project -> FOR_CLIENT -> Company."""
    query = """
        MATCH (e:Employee)-[:WORKED_ON]->(p:Project)-[:FOR_CLIENT]->(c:Company)
        WHERE toLower(c.name) = toLower($client_name)
          AND ($authorized_levels IS NULL OR e.access_level IN $authorized_levels)
        RETURN DISTINCT e.pg_id AS pg_id, e.full_name AS full_name,
               e.employee_code AS employee_code, e.years_experience AS years_experience,
               collect(DISTINCT p.name) AS project_names
    """
    rows = await asyncio.to_thread(
        _run_read, query, client_name=client_name, authorized_levels=authorized_levels
    )
    return [
        EmployeeMatch(
            pg_id=row["pg_id"],
            full_name=row["full_name"],
            employee_code=row["employee_code"],
            years_experience=row["years_experience"],
            detail=", ".join(row["project_names"]),
        )
        for row in rows
    ]


@dataclass
class SkillEntry:
    name: str
    years_experience: int | None
    proficiency: str | None


@dataclass
class CertificationEntry:
    name: str
    issued_date: str | None
    expiry_date: str | None


@dataclass
class RoleEntry:
    title: str
    start_date: str | None
    end_date: str | None


@dataclass
class ProjectEntry:
    name: str
    role_on_project: str | None
    industry: str | None
    client: str | None


@dataclass
class EmployeeProfile:
    """Full employee detail — every field here is a direct graph fact
    (HAS_SKILL/HAS_CERTIFICATION/HAS_ROLE/WORKED_ON relationships), never
    LLM-generated. Used by CV generation (spec section 26): the CV renders
    this structured data directly into a template rather than asking an
    LLM to write prose about the employee, which is what makes "must not
    invent employee experience" an architectural guarantee rather than a
    prompt instruction."""

    pg_id: str
    full_name: str
    employee_code: str
    years_experience: int
    company_name: str | None
    department_name: str | None
    skills: list[SkillEntry]
    certifications: list[CertificationEntry]
    roles: list[RoleEntry]
    projects: list[ProjectEntry]


async def get_employee_profile(
    pg_id: str, authorized_levels: list[str] | None = None
) -> EmployeeProfile | None:
    """`authorized_levels=None` means unrestricted (internal callers only
    — scripts, tests). A real API caller always passes their actual
    authorized levels; an employee that exists but isn't authorized for
    the caller returns None, the same shape as "doesn't exist" — this
    endpoint never reveals that a restricted employee record exists to an
    unauthorized caller, matching the retrieval-time enforcement pattern
    docs/security.md establishes for documents."""
    query = """
        MATCH (e:Employee {pg_id: $pg_id})
        WHERE $authorized_levels IS NULL OR e.access_level IN $authorized_levels
        OPTIONAL MATCH (e)-[:WORKS_FOR]->(comp:Company)
        OPTIONAL MATCH (e)-[:WORKS_IN]->(dept:Department)
        OPTIONAL MATCH (e)-[skill_rel:HAS_SKILL]->(skill:Skill)
        OPTIONAL MATCH (e)-[cert_rel:HAS_CERTIFICATION]->(cert:Certification)
        OPTIONAL MATCH (e)-[role_rel:HAS_ROLE]->(role:JobRole)
        OPTIONAL MATCH (e)-[proj_rel:WORKED_ON]->(proj:Project)
        OPTIONAL MATCH (proj)-[:BELONGS_TO]->(ind:Industry)
        OPTIONAL MATCH (proj)-[:FOR_CLIENT]->(client:Company)
        RETURN e.pg_id AS pg_id, e.full_name AS full_name, e.employee_code AS employee_code,
               e.years_experience AS years_experience,
               comp.name AS company_name, dept.name AS department_name,
               collect(DISTINCT {
                   name: skill.name, years_experience: skill_rel.years_experience,
                   proficiency: skill_rel.proficiency
               }) AS skills,
               collect(DISTINCT {
                   name: cert.name, issued_date: toString(cert_rel.issued_date),
                   expiry_date: toString(cert_rel.expiry_date)
               }) AS certifications,
               collect(DISTINCT {
                   title: role.title, start_date: toString(role_rel.start_date),
                   end_date: toString(role_rel.end_date)
               }) AS roles,
               collect(DISTINCT {
                   name: proj.name, role_on_project: proj_rel.role_on_project,
                   industry: ind.name, client: client.name
               }) AS projects
    """
    rows = await asyncio.to_thread(
        _run_read, query, pg_id=pg_id, authorized_levels=authorized_levels
    )
    if not rows:
        return None
    row = rows[0]
    if row["full_name"] is None:
        return None  # no Employee node with this pg_id

    return EmployeeProfile(
        pg_id=row["pg_id"],
        full_name=row["full_name"],
        employee_code=row["employee_code"],
        years_experience=row["years_experience"],
        company_name=row["company_name"],
        department_name=row["department_name"],
        skills=[
            SkillEntry(s["name"], s["years_experience"], s["proficiency"])
            for s in row["skills"]
            if s["name"] is not None
        ],
        certifications=[
            CertificationEntry(c["name"], c["issued_date"], c["expiry_date"])
            for c in row["certifications"]
            if c["name"] is not None
        ],
        roles=[
            RoleEntry(r["title"], r["start_date"], r["end_date"])
            for r in row["roles"]
            if r["title"] is not None
        ],
        projects=[
            ProjectEntry(p["name"], p["role_on_project"], p["industry"], p["client"])
            for p in row["projects"]
            if p["name"] is not None
        ],
    )


async def project_team(
    project_name: str, authorized_levels: list[str] | None = None
) -> list[EmployeeMatch]:
    """'Who worked on Project X?'"""
    query = """
        MATCH (e:Employee)-[r:WORKED_ON]->(p:Project)
        WHERE toLower(p.name) = toLower($project_name)
          AND ($authorized_levels IS NULL OR e.access_level IN $authorized_levels)
        RETURN e.pg_id AS pg_id, e.full_name AS full_name, e.employee_code AS employee_code,
               e.years_experience AS years_experience, r.role_on_project AS role_on_project
    """
    rows = await asyncio.to_thread(
        _run_read, query, project_name=project_name, authorized_levels=authorized_levels
    )
    return [
        EmployeeMatch(
            pg_id=row["pg_id"],
            full_name=row["full_name"],
            employee_code=row["employee_code"],
            years_experience=row["years_experience"],
            detail=row["role_on_project"],
        )
        for row in rows
    ]
