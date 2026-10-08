"""Requires `docker compose up -d postgres qdrant` running locally, and
migrations applied (`alembic upgrade head`)."""

import shutil
from pathlib import Path

import pytest
from sqlalchemy import select

from app.models.document import Document
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import IngestionError, ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import delete_document_vectors


@pytest.fixture(autouse=True)
async def _cleanup():
    """Tracks document ids created during the test and removes them from
    BOTH PostgreSQL and Qdrant on teardown (always runs, even if the test
    body raises). Ingestion now indexes into Qdrant too (Phase 3) — a
    Postgres-only cleanup here previously left orphaned vectors behind on
    every test run, polluting later searches (caught via a manual /search
    smoke test, not by these tests themselves, since nothing here asserted
    on corpus size)."""
    created_document_ids: list = []
    yield created_document_ids

    factory = get_session_factory()
    async with factory() as session:
        for document_id in created_document_ids:
            delete_document_vectors(document_id)
            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)
        await session.commit()
    invalidate_bm25_index()

    if RAW_STORAGE_ROOT.exists():
        shutil.rmtree(RAW_STORAGE_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_ingest_txt_document_creates_document_and_chunks(_cleanup):
    factory = get_session_factory()
    content = b"This is the first paragraph.\n\nThis is the second paragraph with more words in it."

    async with factory() as session:
        result = await ingest_document(
            session, "policy.txt", content, chunk_size=10, chunk_overlap=0
        )
        _cleanup.append(result.document_id)

        assert result.chunk_count >= 1

        document = await session.get(Document, result.document_id)
        assert document is not None
        assert document.source_filename == "policy.txt"
        assert Path(document.source_uri).exists()


@pytest.mark.asyncio
async def test_ingest_duplicate_content_is_rejected(_cleanup):
    factory = get_session_factory()
    content = b"Duplicate content for testing exact-duplicate detection."

    async with factory() as session:
        first = await ingest_document(session, "dup1.txt", content)
        _cleanup.append(first.document_id)

        with pytest.raises(IngestionError, match="Duplicate"):
            await ingest_document(session, "dup2.txt", content)


@pytest.mark.asyncio
async def test_ingest_empty_file_is_rejected(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        with pytest.raises(IngestionError):
            await ingest_document(session, "empty.txt", b"")

        result = await session.scalars(
            select(Document).where(Document.source_filename == "empty.txt")
        )
        assert result.first() is None
