"""Requires postgres + qdrant running. Verifies spec section 19 (conflict
detection, see app/services/ingestion/conflict_detection.py's module
docstring for the design rationale): two topically-related documents
stating different numeric values for the same kind of quantity are
flagged, without blocking ingestion or discarding either document.
"""

import shutil
import uuid

import pytest

from app.models.document import Document
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import delete_document_vectors


@pytest.fixture(autouse=True)
async def _cleanup():
    created_document_ids: list = []
    yield created_document_ids

    factory = get_session_factory()
    async with factory() as session:
        for document_id in reversed(created_document_ids):
            delete_document_vectors(document_id)
            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)
        await session.commit()
    invalidate_bm25_index()

    if RAW_STORAGE_ROOT.exists():
        shutil.rmtree(RAW_STORAGE_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_different_values_for_the_same_topic_are_flagged(_cleanup):
    topic = f"Qorvanite{uuid.uuid4().hex[:8]}"

    factory = get_session_factory()
    async with factory() as session:
        first = await ingest_document(
            session,
            "policy_a.txt",
            f"The {topic} exposure limit must not exceed 60 units per hour.".encode(),
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup.append(first.document_id)
        assert first.conflicting_claim_count == 0  # nothing else in the corpus yet

        second = await ingest_document(
            session,
            "policy_b.txt",
            f"Per the updated guidance, the {topic} exposure limit must not "
            f"exceed 40 units per hour.".encode(),
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup.append(second.document_id)

        assert second.conflicting_claim_count >= 1

        # Detection only — both documents must still exist.
        doc1 = await session.get(Document, first.document_id)
        doc2 = await session.get(Document, second.document_id)
        assert doc1 is not None
        assert doc2 is not None


@pytest.mark.asyncio
async def test_same_value_restated_is_not_flagged_as_conflicting(_cleanup):
    topic = f"Blorvex{uuid.uuid4().hex[:8]}"

    factory = get_session_factory()
    async with factory() as session:
        first = await ingest_document(
            session,
            "memo_a.txt",
            f"The {topic} threshold must not exceed 25 meters.".encode(),
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup.append(first.document_id)

        second = await ingest_document(
            session,
            "memo_b.txt",
            f"As a reminder, the {topic} threshold must not exceed 25 meters.".encode(),
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup.append(second.document_id)

        # Same number, same unit, same topic — agreement, not conflict.
        assert second.conflicting_claim_count == 0


@pytest.mark.asyncio
async def test_unrelated_documents_with_different_units_are_not_flagged(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        first = await ingest_document(
            session,
            "unrelated_a.txt",
            b"The warehouse closes at 60 minutes past business hours on Fridays.",
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup.append(first.document_id)

        second = await ingest_document(
            session,
            "unrelated_b.txt",
            b"The cafeteria serves lunch for 60 days each quarter on a rotating menu.",
            chunk_size=40,
            chunk_overlap=0,
        )
        _cleanup.append(second.document_id)

        # Same number (60) but DIFFERENT units (minutes vs. days) and
        # unrelated topics — must not be flagged just because a number
        # happens to match.
        assert second.conflicting_claim_count == 0
