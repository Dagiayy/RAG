import json
from types import SimpleNamespace

import pytest

from app.services.generation import answer_generation as ag_module
from app.services.generation.answer_generation import generate_answer
from app.services.retrieval.hybrid import HybridResult


def _doc_result(chunk_id: str, text: str) -> HybridResult:
    return HybridResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        fused_score=0.9,
        vector_score=0.9,
        bm25_score=None,
        text=text,
        source_filename="test.txt",
        page_number=1,
        section=None,
    )


class _FakeClient:
    """Mimics openai.AsyncOpenAI's chat.completions.create() surface,
    returning a fixed JSON string so we can deterministically test the
    citation-safety filtering — a real LLM won't reliably hallucinate a
    specific bad marker on demand, but this invariant must hold regardless
    of what the model outputs."""

    def __init__(self, content: str | None = None, exc: Exception | None = None):
        self._content = content
        self._exc = exc

    async def _create(self, **kwargs):
        if self._exc:
            raise self._exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


@pytest.mark.asyncio
async def test_no_evidence_skips_llm_call_entirely(monkeypatch):
    called = False

    def _fail_if_called():
        nonlocal called
        called = True
        raise AssertionError("LLM should never be called with zero evidence")

    monkeypatch.setattr(ag_module, "get_llm_client", _fail_if_called)

    result = await generate_answer("anything", [], [])
    assert called is False
    assert result.confidence == "insufficient_evidence"
    assert result.citations == []


@pytest.mark.asyncio
async def test_citations_never_include_markers_the_llm_hallucinates(monkeypatch):
    # LLM claims to have cited marker 99, which was never given to it —
    # only marker 1 was real evidence.
    fake_content = json.dumps(
        {
            "answer": "Some answer [1] [99].",
            "confidence": "evidence_supported",
            "cited_markers": [1, 99],
        }
    )
    monkeypatch.setattr(ag_module, "get_llm_client", lambda: _FakeClient(content=fake_content))

    result = await generate_answer("query", [], [_doc_result("c1", "real evidence text")])

    assert [c.marker for c in result.citations] == [1]
    assert 99 not in [c.marker for c in result.citations]


@pytest.mark.asyncio
async def test_invalid_json_from_llm_falls_back_to_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(ag_module, "get_llm_client", lambda: _FakeClient(content="not valid json"))

    result = await generate_answer("query", [], [_doc_result("c1", "real evidence text")])

    assert result.confidence == "insufficient_evidence"
    assert result.answer == ag_module.INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.citations == []


@pytest.mark.asyncio
async def test_invalid_schema_from_llm_falls_back_to_insufficient_evidence(monkeypatch):
    # missing required "confidence" field
    fake_content = json.dumps({"answer": "An answer.", "cited_markers": [1]})
    monkeypatch.setattr(ag_module, "get_llm_client", lambda: _FakeClient(content=fake_content))

    result = await generate_answer("query", [], [_doc_result("c1", "real evidence text")])

    assert result.confidence == "insufficient_evidence"


@pytest.mark.asyncio
async def test_llm_connection_error_falls_back_to_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(
        ag_module, "get_llm_client", lambda: _FakeClient(exc=ConnectionError("simulated"))
    )

    result = await generate_answer("query", [], [_doc_result("c1", "real evidence text")])

    assert result.confidence == "insufficient_evidence"
    assert result.answer == ag_module.INSUFFICIENT_EVIDENCE_MESSAGE


@pytest.mark.asyncio
async def test_valid_response_passes_through_correctly(monkeypatch):
    fake_content = json.dumps(
        {"answer": "The answer is X [1].", "confidence": "evidence_supported", "cited_markers": [1]}
    )
    monkeypatch.setattr(ag_module, "get_llm_client", lambda: _FakeClient(content=fake_content))

    result = await generate_answer("query", [], [_doc_result("c1", "real evidence text")])

    assert result.answer == "The answer is X [1]."
    assert result.confidence == "evidence_supported"
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert result.citations[0].source_filename == "test.txt"
