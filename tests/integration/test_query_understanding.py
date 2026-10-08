"""Requires Ollama running locally with the configured generation model
pulled (see .env OPENAI_MODEL). Exercises real LLM structured
extraction — no mocking, since the whole point of Phase 7 is verifying the
schema-constrained extraction actually works against a real small local
model, not just that our Pydantic validation logic works. Self-skips (not
a fake pass) when Ollama isn't reachable, e.g. in CI.
"""

import pytest

from app.services.generation.query_understanding import understand_query
from tests.conftest import ollama_is_reachable

pytestmark = pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")


@pytest.mark.asyncio
async def test_employee_search_with_skill_extracted():
    intent = await understand_query("Which employees have GIS Mapping experience?")
    assert intent.intent == "employee_search"
    assert intent.entities.get("skill", "").lower() == "gis mapping"


@pytest.mark.asyncio
async def test_multi_hop_skill_industry_and_filter_extracted():
    intent = await understand_query(
        "Which employees with Survey Design skills worked on Humanitarian Aid "
        "projects with more than 5 years experience?"
    )
    assert intent.intent == "employee_search"
    assert "skill" in intent.entities
    assert "industry" in intent.entities
    assert any(f.field == "years_experience" and f.operator == ">" for f in intent.filters)


@pytest.mark.asyncio
async def test_project_lookup_extracts_project_name():
    intent = await understand_query("Who worked on the Refugee Camp Needs Assessment project?")
    assert intent.intent == "project_lookup"
    assert "refugee camp needs assessment" in intent.entities.get("project", "").lower()


@pytest.mark.asyncio
async def test_document_question_classified_as_document_qa():
    intent = await understand_query("What does the confined space entry safety procedure require?")
    assert intent.intent == "document_qa"
    assert intent.entities == {}


@pytest.mark.asyncio
async def test_unrelated_question_falls_back_to_general_qa():
    intent = await understand_query("What's a good recipe for banana bread?")
    assert intent.intent == "general_qa"


@pytest.mark.asyncio
async def test_unreachable_llm_degrades_to_general_qa(monkeypatch):
    from app.services.generation import query_understanding as qu_module

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise ConnectionError("simulated: LLM unreachable")

    class _FailingChat:
        completions = _FailingCompletions()

    class _FailingClient:
        chat = _FailingChat()

    monkeypatch.setattr(qu_module, "get_llm_client", lambda: _FailingClient())

    intent = await understand_query("Which employees have GIS Mapping experience?")
    assert intent.intent == "general_qa"
