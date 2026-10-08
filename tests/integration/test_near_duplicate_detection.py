"""Requires postgres + qdrant running. Verifies spec section 18 (near-
duplicate detection): substantially-similar-but-not-identical content
across documents is detected (and reported), without blocking ingestion or
discarding either document — provenance of both is preserved.
"""

import shutil

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
async def test_near_identical_content_across_documents_is_detected(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        first = await ingest_document(
            session,
            "safety_memo_draft.txt",
            b"All employees must complete confined space entry training before June 30.",
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(first.document_id)
        assert first.near_duplicate_count == 0  # nothing else in the corpus yet

        # Same content, trivially reworded (not byte-identical, so the
        # exact-checksum duplicate check does NOT catch this).
        second = await ingest_document(
            session,
            "safety_memo_final.txt",
            b"All staff members must complete confined space entry training before June 30th.",
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(second.document_id)

        assert second.near_duplicate_count >= 1

        # Both documents must still exist — detection never discards
        # either side (provenance preserved).
        doc1 = await session.get(Document, first.document_id)
        doc2 = await session.get(Document, second.document_id)
        assert doc1 is not None
        assert doc2 is not None


@pytest.mark.asyncio
async def test_unrelated_content_is_not_flagged_as_near_duplicate(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        first = await ingest_document(
            session,
            "safety_memo.txt",
            b"All employees must complete confined space entry training before June 30.",
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(first.document_id)

        second = await ingest_document(
            session,
            "cafeteria_menu.txt",
            b"The cafeteria will serve pasta on Tuesdays and grilled chicken on Fridays.",
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(second.document_id)

        assert second.near_duplicate_count == 0
