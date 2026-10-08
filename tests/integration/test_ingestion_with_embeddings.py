"""Requires postgres + qdrant running, and downloads the real embedding
model on first run (sentence-transformers/all-MiniLM-L6-v2, ~90MB)."""

import shutil

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.document import Document
from app.repositories.db import get_session_factory
from app.services.embeddings.service import get_embedding_service
from app.services.ingestion.pipeline import ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.vector import delete_document_vectors, search


@pytest.fixture(autouse=True)
async def _cleanup():
    """Runs on teardown regardless of whether the test body raised, so a
    failing assertion never leaves an orphaned Document row / Qdrant
    vectors / raw file behind for the next run to trip over."""
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

    if RAW_STORAGE_ROOT.exists():
        shutil.rmtree(RAW_STORAGE_ROOT, ignore_errors=True)


@pytest.mark.asyncio
async def test_ingest_document_indexes_chunks_into_qdrant(_cleanup):
    factory = get_session_factory()
    content = (
        b"Siemens PLCs are widely used in cement plant automation.\n\n"
        b"Unrelated paragraph about quarterly financial reporting."
    )

    async with factory() as session:
        result = await ingest_document(
            session, "plc_notes.txt", content, chunk_size=20, chunk_overlap=0
        )
        _cleanup.append(result.document_id)

        document = await session.scalar(
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == result.document_id)
        )
        assert document.chunks, "expected at least one chunk to be persisted"
        for chunk in document.chunks:
            assert chunk.embedding_model == get_settings().embedding_model
            assert chunk.embedding_model_version == get_settings().embedding_model_version

    query_vector = get_embedding_service().encode_one("Siemens PLC automation")
    results = search(query_vector, top_k=5, filters={"document_id": str(result.document_id)})

    assert len(results) >= 1
    assert any("Siemens" in r.payload["text"] for r in results)
