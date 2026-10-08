"""End-to-end coverage for the Phase 9/11 synthetic-document scenarios
(see scripts/seed_synthetic_documents.py's module docstring) through the
REAL retrieval pipeline — embedding, Qdrant, BM25, RRF fusion, reranking,
and generation — not hand-built evidence objects like
tests/integration/test_answer_generation.py uses. Each test ingests its
own isolated, uuid-tagged documents (same pattern as
tests/security/test_retrieval_leakage.py) so this doesn't depend on
scripts/seed_synthetic_documents.py having been run, and cleans up fully.

Requires postgres + qdrant + Ollama running. Self-skips when Ollama isn't
reachable.
"""

import shutil
import uuid

import pytest

from app.models.document import AccessLevel, Document
from app.models.user import User
from app.pipelines.query_pipeline import run_query
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import delete_document_vectors
from tests.conftest import ollama_is_reachable

pytestmark = pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {"document_ids": [], "user_ids": []}
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for document_id in reversed(created["document_ids"]):
            delete_document_vectors(document_id)
            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)
        for user_id in created["user_ids"]:
            user = await session.get(User, user_id)
            if user is not None:
                await session.delete(user)
        await session.commit()
    invalidate_bm25_index()

    if RAW_STORAGE_ROOT.exists():
        shutil.rmtree(RAW_STORAGE_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_two_independent_documents_with_different_values_yield_conflicting_evidence(
    _cleanup,
):
    topic = f"Zyntrevium{uuid.uuid4().hex[:8]}"

    factory = get_session_factory()
    async with factory() as session:
        addendum = await ingest_document(
            session,
            "addendum.txt",
            f"Per the updated safety addendum, {topic} exposure limits must not "
            f"exceed 40 units per hour.".encode(),
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup["document_ids"].append(addendum.document_id)

        original = await ingest_document(
            session,
            "original_policy.txt",
            f"The original safety policy states {topic} exposure limits must not "
            f"exceed 60 units per hour.".encode(),
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup["document_ids"].append(original.document_id)

        result = await run_query(session, f"What is the exposure limit for {topic}?")

    assert result.plan.use_document_rag is True
    assert result.generated.confidence == "conflicting_evidence"
    assert "40" in result.generated.answer
    assert "60" in result.generated.answer
    cited_ids = {c.document_id for c in result.generated.citations}
    assert str(addendum.document_id) in cited_ids
    assert str(original.document_id) in cited_ids


@pytest.mark.asyncio
async def test_access_ladder_each_level_retrievable_only_at_or_above_its_clearance(_cleanup):
    """Seeds one document at each AccessLevel (mirrors
    scripts/seed_synthetic_documents.py's access-level ladder) and
    confirms real hybrid retrieval respects clearance filtering across
    the FULL ladder, not just the single PUBLIC/RESTRICTED pair already
    covered by tests/security/test_retrieval_leakage.py."""
    from app.models.user import ClearanceLevel
    from app.services.retrieval.hybrid import run_search

    topic = f"Qorvanite{uuid.uuid4().hex[:8]}"
    levels = [
        AccessLevel.PUBLIC,
        AccessLevel.INTERNAL,
        AccessLevel.DEPARTMENT,
        AccessLevel.CONFIDENTIAL,
        AccessLevel.RESTRICTED,
    ]

    factory = get_session_factory()
    async with factory() as session:
        doc_ids_by_level = {}
        for level in levels:
            result = await ingest_document(
                session,
                f"{level.value}_doc.txt",
                f"The {topic} figure at the {level.value} level is a specific "
                f"confidential number.".encode(),
                access_level=level,
                chunk_size=40,
                chunk_overlap=0,
            )
            _cleanup["document_ids"].append(result.document_id)
            doc_ids_by_level[level] = str(result.document_id)

        # A DEPARTMENT-clearance user should see PUBLIC/INTERNAL/DEPARTMENT
        # content but not CONFIDENTIAL/RESTRICTED.
        user = User(
            username=f"ladder-test-{uuid.uuid4().hex[:8]}",
            clearance_level=ClearanceLevel.DEPARTMENT,
        )
        session.add(user)
        await session.flush()
        _cleanup["user_ids"].append(user.id)
        await session.commit()

        from app.core.security.access import authorized_access_levels

        filters = {"access_level": authorized_access_levels(user.clearance_level)}
        results = await run_search(session, topic, top_k=20, filters=filters, rerank=False)

    retrieved_document_ids = {r.document_id for r in results}
    assert doc_ids_by_level[AccessLevel.PUBLIC] in retrieved_document_ids
    assert doc_ids_by_level[AccessLevel.INTERNAL] in retrieved_document_ids
    assert doc_ids_by_level[AccessLevel.DEPARTMENT] in retrieved_document_ids
    assert doc_ids_by_level[AccessLevel.CONFIDENTIAL] not in retrieved_document_ids
    assert doc_ids_by_level[AccessLevel.RESTRICTED] not in retrieved_document_ids
