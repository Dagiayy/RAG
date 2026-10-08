"""Seeds a small, realistic synthetic enterprise dataset into PostgreSQL and
syncs it into Neo4j — enough to exercise Phase 6/7's graph retrieval
end-to-end. Not Phase 9's full synthetic dataset (which adds conflicting
document versions, duplicates, restricted docs, etc.) — just the entity
graph (employees/skills/projects/clients).

Domain flavor is loosely inspired by real-world field-data-collection /
monitoring-evaluation-and-learning (MEL) / IoT-for-development firms (the
kind of organization that does drought monitoring, refugee needs
assessments, agricultural IoT sensor rollouts, and ESG verification work)
rather than a generic manufacturing-engineering example — but every
company, employee, and project name here is fictional. No real
organization's data or any real person's identity is used.

Run: python scripts/seed_demo_data.py
Idempotent: re-running upserts by unique name/code rather than duplicating.
"""

import asyncio
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.services.graph.sync import sync_all_to_graph

OWN_ORG_NAME = "Meridian Field Data & Monitoring Services"


async def _get_or_create(session: AsyncSession, model, unique_field: str, **fields):
    stmt = select(model).where(getattr(model, unique_field) == fields[unique_field])
    existing = await session.scalar(stmt)
    if existing:
        return existing
    obj = model(**fields)
    session.add(obj)
    await session.flush()
    return obj


async def seed(session: AsyncSession) -> None:
    own_org = await _get_or_create(session, Company, "name", name=OWN_ORG_NAME, is_own_org=True)

    clients = {
        name: await _get_or_create(session, Company, "name", name=name, is_own_org=False)
        for name in [
            "Global Development Fund",
            "AgriRise Foundation",
            "Horn Relief International",
            "Continental Development Bank",
        ]
    }

    departments = {
        name: await _get_or_create(session, Department, "name", name=name)
        for name in [
            "Field Operations",
            "Data Analysis & Analytics",
            "Monitoring, Evaluation & Learning",
            "Verification & Compliance",
            "IoT & Sensor Engineering",
        ]
    }

    industries = {
        name: await _get_or_create(session, Industry, "name", name=name)
        for name in ["Agriculture", "Humanitarian Aid", "Environmental Monitoring"]
    }

    skills = {
        name: await _get_or_create(session, Skill, "name", name=name, category=category)
        for name, category in [
            ("Field Data Collection (ODK/KoBoToolbox)", "data_collection"),
            ("Statistical Analysis (R/Python)", "analytics"),
            ("GIS Mapping", "geospatial"),
            ("LoRaWAN/IoT Sensors", "iot"),
            ("Remote Sensing", "geospatial"),
            ("Survey Design", "research"),
            ("ESG Verification", "compliance"),
            ("Qualitative Research", "research"),
            ("Weather Station Deployment", "iot"),
        ]
    }

    certifications = {
        name: await _get_or_create(session, Certification, "name", name=name, issuing_body=body)
        for name, body in [
            ("Certified M&E Professional", "International Program Evaluation Network"),
            ("GIS Professional Certification", "GIS Certification Institute"),
            ("ESG Assurance Certification", "Global ESG Assurance Board"),
        ]
    }

    job_roles = {
        title: await _get_or_create(session, JobRole, "title", title=title)
        for title in [
            "Field Data Collector",
            "Data Analyst",
            "M&E Specialist",
            "Verification Officer",
            "IoT/Sensor Engineer",
            "Project Manager",
            "GIS Analyst",
        ]
    }

    projects = {
        "Drought Resilience Monitoring": await _get_or_create(
            session,
            Project,
            "name",
            name="Drought Resilience Monitoring",
            company_id=own_org.id,
            client_id=clients["AgriRise Foundation"].id,
            industry_id=industries["Agriculture"].id,
            start_date=date(2024, 2, 1),
            description=(
                "Continuous drought indicator monitoring across smallholder farming regions."
            ),
        ),
        "Refugee Camp Needs Assessment": await _get_or_create(
            session,
            Project,
            "name",
            name="Refugee Camp Needs Assessment",
            company_id=own_org.id,
            client_id=clients["Horn Relief International"].id,
            industry_id=industries["Humanitarian Aid"].id,
            start_date=date(2024, 5, 1),
            description="Rapid needs assessment survey across three refugee camp sites.",
        ),
        "Smallholder Irrigation IoT Rollout": await _get_or_create(
            session,
            Project,
            "name",
            name="Smallholder Irrigation IoT Rollout",
            company_id=own_org.id,
            client_id=clients["Global Development Fund"].id,
            industry_id=industries["Agriculture"].id,
            start_date=date(2023, 11, 1),
            description=(
                "LoRaWAN soil-moisture sensor deployment for smallholder irrigation scheduling."
            ),
        ),
        "ESG Compliance Verification Program": await _get_or_create(
            session,
            Project,
            "name",
            name="ESG Compliance Verification Program",
            company_id=own_org.id,
            client_id=clients["Continental Development Bank"].id,
            industry_id=industries["Environmental Monitoring"].id,
            start_date=date(2024, 1, 15),
            description=(
                "Independent ESG compliance verification for bank-financed infrastructure projects."
            ),
        ),
    }

    employee_specs = [
        dict(
            employee_code="MER-001",
            full_name="Amara Bekele",
            email="amara.bekele@example-meridian.org",
            hire_date=date(2019, 3, 1),
            years_experience=5,
            department="Field Operations",
            role="Field Data Collector",
            skills=[
                ("Field Data Collection (ODK/KoBoToolbox)", 5, "expert"),
                ("Survey Design", 4, "intermediate"),
            ],
            certifications=[],
            projects=[
                ("Drought Resilience Monitoring", "Field Data Collector"),
                ("Refugee Camp Needs Assessment", "Field Data Collector"),
            ],
        ),
        dict(
            employee_code="MER-002",
            full_name="Daniel Tesfaye",
            email="daniel.tesfaye@example-meridian.org",
            hire_date=date(2018, 6, 1),
            years_experience=6,
            department="Data Analysis & Analytics",
            role="Data Analyst",
            skills=[
                ("Statistical Analysis (R/Python)", 6, "expert"),
                ("GIS Mapping", 3, "intermediate"),
            ],
            certifications=[("GIS Professional Certification", date(2021, 4, 1), date(2027, 4, 1))],
            projects=[("Drought Resilience Monitoring", "Data Analyst")],
        ),
        dict(
            employee_code="MER-003",
            full_name="Sara Mensah",
            email="sara.mensah@example-meridian.org",
            hire_date=date(2017, 1, 15),
            years_experience=7,
            department="Monitoring, Evaluation & Learning",
            role="M&E Specialist",
            skills=[("Survey Design", 7, "expert"), ("Qualitative Research", 5, "expert")],
            certifications=[("Certified M&E Professional", date(2020, 9, 1), date(2026, 9, 1))],
            projects=[("Refugee Camp Needs Assessment", "M&E Lead")],
        ),
        dict(
            employee_code="MER-004",
            full_name="Yonas Alemu",
            email="yonas.alemu@example-meridian.org",
            hire_date=date(2020, 8, 1),
            years_experience=4,
            department="IoT & Sensor Engineering",
            role="IoT/Sensor Engineer",
            skills=[
                ("LoRaWAN/IoT Sensors", 4, "expert"),
                ("Weather Station Deployment", 3, "intermediate"),
            ],
            certifications=[],
            projects=[("Smallholder Irrigation IoT Rollout", "IoT Engineer")],
        ),
        dict(
            employee_code="MER-005",
            full_name="Fatima Noor",
            email="fatima.noor@example-meridian.org",
            hire_date=date(2019, 11, 1),
            years_experience=5,
            department="Verification & Compliance",
            role="Verification Officer",
            skills=[("ESG Verification", 5, "expert")],
            certifications=[("ESG Assurance Certification", date(2022, 2, 1), date(2027, 2, 1))],
            projects=[("ESG Compliance Verification Program", "Lead Verifier")],
        ),
        dict(
            employee_code="MER-006",
            full_name="Michael Owusu",
            email="michael.owusu@example-meridian.org",
            hire_date=date(2017, 7, 1),
            years_experience=6,
            department="Data Analysis & Analytics",
            role="GIS Analyst",
            skills=[("GIS Mapping", 6, "expert"), ("Remote Sensing", 4, "intermediate")],
            certifications=[("GIS Professional Certification", date(2019, 5, 1), date(2025, 5, 1))],
            projects=[
                ("Smallholder Irrigation IoT Rollout", "GIS Analyst"),
                ("Drought Resilience Monitoring", "GIS Analyst"),
            ],
        ),
        dict(
            employee_code="MER-007",
            full_name="Grace Achieng",
            email="grace.achieng@example-meridian.org",
            hire_date=date(2015, 4, 1),
            years_experience=9,
            department="Field Operations",
            role="Project Manager",
            skills=[("Survey Design", 8, "expert")],
            certifications=[],
            projects=[
                ("Drought Resilience Monitoring", "Project Manager"),
                ("Refugee Camp Needs Assessment", "Project Manager"),
                ("Smallholder Irrigation IoT Rollout", "Project Manager"),
                ("ESG Compliance Verification Program", "Project Manager"),
            ],
        ),
        dict(
            employee_code="MER-008",
            full_name="Samuel Kimani",
            email="samuel.kimani@example-meridian.org",
            hire_date=date(2022, 2, 1),
            years_experience=3,
            department="Field Operations",
            role="Field Data Collector",
            skills=[("Field Data Collection (ODK/KoBoToolbox)", 3, "intermediate")],
            certifications=[],
            projects=[("Smallholder Irrigation IoT Rollout", "Field Data Collector")],
        ),
    ]

    for spec in employee_specs:
        employee = await _get_or_create(
            session,
            Employee,
            "employee_code",
            employee_code=spec["employee_code"],
            full_name=spec["full_name"],
            email=spec["email"],
            hire_date=spec["hire_date"],
            years_experience=spec["years_experience"],
            department_id=departments[spec["department"]].id,
            company_id=own_org.id,
        )

        for skill_name, years, proficiency in spec["skills"]:
            existing = await session.scalar(
                select(EmployeeSkill).where(
                    EmployeeSkill.employee_id == employee.id,
                    EmployeeSkill.skill_id == skills[skill_name].id,
                )
            )
            if not existing:
                session.add(
                    EmployeeSkill(
                        employee_id=employee.id,
                        skill_id=skills[skill_name].id,
                        years_experience=years,
                        proficiency=proficiency,
                    )
                )

        for cert_name, issued, expiry in spec["certifications"]:
            existing = await session.scalar(
                select(EmployeeCertification).where(
                    EmployeeCertification.employee_id == employee.id,
                    EmployeeCertification.certification_id == certifications[cert_name].id,
                )
            )
            if not existing:
                session.add(
                    EmployeeCertification(
                        employee_id=employee.id,
                        certification_id=certifications[cert_name].id,
                        issued_date=issued,
                        expiry_date=expiry,
                    )
                )

        existing_role = await session.scalar(
            select(EmployeeRole).where(
                EmployeeRole.employee_id == employee.id,
                EmployeeRole.job_role_id == job_roles[spec["role"]].id,
            )
        )
        if not existing_role:
            session.add(
                EmployeeRole(
                    employee_id=employee.id,
                    job_role_id=job_roles[spec["role"]].id,
                    start_date=spec["hire_date"],
                )
            )

        for project_name, role_on_project in spec["projects"]:
            existing_proj = await session.scalar(
                select(EmployeeProject).where(
                    EmployeeProject.employee_id == employee.id,
                    EmployeeProject.project_id == projects[project_name].id,
                )
            )
            if not existing_proj:
                session.add(
                    EmployeeProject(
                        employee_id=employee.id,
                        project_id=projects[project_name].id,
                        role_on_project=role_on_project,
                    )
                )

    await session.commit()


async def main() -> None:
    factory = get_session_factory()
    async with factory() as session:
        print("Seeding PostgreSQL...")
        await seed(session)
        print("Seeded. Syncing to Neo4j...")
        summary = await sync_all_to_graph(session)
        print(
            f"Graph sync complete: {summary.employees} employees, {summary.projects} projects, "
            f"{summary.companies} companies, {summary.skills} skills, "
            f"{summary.relationships} relationships."
        )


if __name__ == "__main__":
    asyncio.run(main())
