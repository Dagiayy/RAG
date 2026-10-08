"""Requires postgres + neo4j running (+ Ollama for the bulk-cv tests,
which self-skip if unreachable). Uses the real FastAPI app end-to-end
(auth, graph, rendering), not mocks.
"""

import uuid
import zipfile
from io import BytesIO

import httpx
import pytest

from app.main import app
from app.models.enterprise import Employee, EmployeeSkill, Skill
from app.models.user import ClearanceLevel, User
from app.repositories.db import get_session_factory
from app.services.graph.client import get_neo4j_driver
from app.services.graph.sync import sync_all_to_graph
from tests.conftest import ollama_is_reachable


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {"employee_ids": [], "skill_ids": [], "user_ids": []}
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for model, key in [(Employee, "employee_ids"), (Skill, "skill_ids"), (User, "user_ids")]:
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


async def _seed_employee_with_skill(session, skill_name: str, cleanup: dict) -> Employee:
    skill = Skill(name=skill_name, category="test")
    session.add(skill)
    await session.flush()
    cleanup["skill_ids"].append(skill.id)

    employee = Employee(
        employee_code=_unique("EMP"), full_name="API Test Employee", years_experience=4
    )
    session.add(employee)
    await session.flush()
    cleanup["employee_ids"].append(employee.id)

    session.add(
        EmployeeSkill(
            employee_id=employee.id, skill_id=skill.id, years_experience=4, proficiency="expert"
        )
    )
    await session.commit()
    await sync_all_to_graph(session)
    return employee


async def _create_user(session, cleanup: dict) -> str:
    user = User(username=_unique("apitest"), clearance_level=ClearanceLevel.INTERNAL)
    session.add(user)
    await session.flush()
    cleanup["user_ids"].append(user.id)
    await session.commit()
    return user.api_key


@pytest.mark.asyncio
async def test_get_employee_profile_over_http(_cleanup):
    skill_name = _unique("Skill")
    factory = get_session_factory()
    async with factory() as session:
        employee = await _seed_employee_with_skill(session, skill_name, _cleanup)
        token = await _create_user(session, _cleanup)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/employees/{employee.id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["full_name"] == "API Test Employee"
        assert any(s["name"] == skill_name for s in body["skills"])

        response = await client.get(f"/employees/{employee.id}")
        assert response.status_code == 401

        response = await client.get(
            "/employees/nonexistent-id", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_generate_single_cv_over_http(_cleanup):
    skill_name = _unique("Skill")
    factory = get_session_factory()
    async with factory() as session:
        employee = await _seed_employee_with_skill(session, skill_name, _cleanup)
        token = await _create_user(session, _cleanup)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/employees/{employee.id}/cv?format=docx",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(response.content) > 0

        response = await client.post(f"/employees/{employee.id}/cv")
        assert response.status_code == 401

        response = await client.post(
            "/employees/nonexistent-id/cv", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404


@pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")
@pytest.mark.asyncio
async def test_bulk_cv_generation_over_http(_cleanup):
    skill_name = _unique("Skill")
    factory = get_session_factory()
    async with factory() as session:
        await _seed_employee_with_skill(session, skill_name, _cleanup)
        token = await _create_user(session, _cleanup)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30.0
    ) as client:
        response = await client.post(
            "/employees/bulk-cv",
            json={
                "query": f"Generate CVs for employees with the skill '{skill_name}'",
                "format": "docx",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            names = zf.namelist()
            assert len(names) == 1
            assert "API_Test_Employee" in names[0]

        response = await client.post(
            "/employees/bulk-cv",
            json={
                "query": "Generate CVs for employees with underwater basket weaving skills",
                "format": "docx",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
