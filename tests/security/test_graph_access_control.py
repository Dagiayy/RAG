"""Security test: clearance-based access control for the graph-backed
surface (/graph/*, /employees/*, and /query's graph-matches path) — added
alongside the Employee.access_level column (see docs/implementation-status.md,
"Post-Phase-11 work"). Same enforcement pattern as
tests/security/test_retrieval_leakage.py for documents: authorization is
computed server-side from the caller's clearance and applied inside the
Cypher query itself (app/services/graph/queries.py), never by post-hoc
filtering or trusting the client. Requires postgres + neo4j running (+
Ollama for the /query test, which self-skips if unreachable); uses the
real FastAPI app end-to-end, not mocks.
"""

import uuid

import httpx
import pytest

from app.main import app
from app.models.document import AccessLevel
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


async def _seed_restricted_employee(session, skill_name: str, cleanup: dict) -> Employee:
    skill = Skill(name=skill_name, category="test")
    session.add(skill)
    await session.flush()
    cleanup["skill_ids"].append(skill.id)

    employee = Employee(
        employee_code=_unique("EMP"),
        full_name="Restricted Test Employee",
        years_experience=5,
        access_level=AccessLevel.RESTRICTED,
    )
    session.add(employee)
    await session.flush()
    cleanup["employee_ids"].append(employee.id)

    session.add(
        EmployeeSkill(
            employee_id=employee.id, skill_id=skill.id, years_experience=5, proficiency="expert"
        )
    )
    await session.commit()
    await sync_all_to_graph(session)
    return employee


async def _create_user(session, clearance: ClearanceLevel, cleanup: dict) -> str:
    user = User(username=_unique("gatest"), clearance_level=clearance)
    session.add(user)
    await session.flush()
    cleanup["user_ids"].append(user.id)
    await session.commit()
    return user.api_key


@pytest.mark.asyncio
async def test_graph_endpoint_requires_authentication():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/graph/employees/by-skill", params={"skill": "anything"})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_low_clearance_user_cannot_see_restricted_employee_via_graph(_cleanup):
    skill_name = _unique("Skill")
    factory = get_session_factory()
    async with factory() as session:
        employee = await _seed_restricted_employee(session, skill_name, _cleanup)
        low_token = await _create_user(session, ClearanceLevel.PUBLIC, _cleanup)
        high_token = await _create_user(session, ClearanceLevel.RESTRICTED, _cleanup)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/graph/employees/by-skill",
            params={"skill": skill_name},
            headers={"Authorization": f"Bearer {low_token}"},
        )
        assert response.status_code == 200
        assert response.json()["results"] == []

        response = await client.get(
            "/graph/employees/by-skill",
            params={"skill": skill_name},
            headers={"Authorization": f"Bearer {high_token}"},
        )
        assert response.status_code == 200
        pg_ids = {r["pg_id"] for r in response.json()["results"]}
        assert str(employee.id) in pg_ids


@pytest.mark.asyncio
async def test_employee_profile_and_cv_respect_clearance(_cleanup):
    skill_name = _unique("Skill")
    factory = get_session_factory()
    async with factory() as session:
        employee = await _seed_restricted_employee(session, skill_name, _cleanup)
        low_token = await _create_user(session, ClearanceLevel.PUBLIC, _cleanup)
        high_token = await _create_user(session, ClearanceLevel.RESTRICTED, _cleanup)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Low clearance: 404, same as a nonexistent employee — existence
        # is not leaked.
        response = await client.get(
            f"/employees/{employee.id}", headers={"Authorization": f"Bearer {low_token}"}
        )
        assert response.status_code == 404

        response = await client.post(
            f"/employees/{employee.id}/cv", headers={"Authorization": f"Bearer {low_token}"}
        )
        assert response.status_code == 404

        # High clearance: both succeed.
        response = await client.get(
            f"/employees/{employee.id}", headers={"Authorization": f"Bearer {high_token}"}
        )
        assert response.status_code == 200
        assert response.json()["full_name"] == "Restricted Test Employee"

        response = await client.post(
            f"/employees/{employee.id}/cv", headers={"Authorization": f"Bearer {high_token}"}
        )
        assert response.status_code == 200


@pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")
@pytest.mark.asyncio
async def test_query_graph_matches_respect_clearance(_cleanup):
    skill_name = _unique("Skill")
    factory = get_session_factory()
    async with factory() as session:
        employee = await _seed_restricted_employee(session, skill_name, _cleanup)
        low_token = await _create_user(session, ClearanceLevel.PUBLIC, _cleanup)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30.0
    ) as client:
        response = await client.post(
            "/query",
            json={"query": f"Which employees have the skill named '{skill_name}'?"},
            headers={"Authorization": f"Bearer {low_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["graph_matches"] == []
        assert str(employee.id) not in {
            c.get("employee_pg_id") for c in body["answer"]["citations"]
        }
