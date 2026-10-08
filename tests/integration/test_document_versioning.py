"""Requires postgres + qdrant running. Verifies spec section 20 (document
versioning): superseding a document updates the old version's status in
BOTH PostgreSQL and Qdrant, and retrieval prefers the current version by
default while still allowing historical queries explicitly.
"""

import shutil

import pytest

from app.models.document import Document, DocumentStatus
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import IngestionError, ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.hybrid import run_search
from app.services.retrieval.vector import delete_document_vectors


@pytest.fixture(autouse=True)
async def _cleanup():
    created_document_ids: list = []
    yield created_document_ids

    factory = get_session_factory()
    async with factory() as session:
        # reverse order: a newer version's supersedes_id FK references the
        # older version, so the older (parent) row can't be deleted first.
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
async def test_superseding_marks_old_version_and_links_chain(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        v1 = await ingest_document(
            session,
            "policy_v1.txt",
            b"Pressure limit ABC123XYZ is 10 bar per policy v1.",
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(v1.document_id)

        v2 = await ingest_document(
            session,
            "policy_v2.txt",
            b"Pressure limit ABC123XYZ is 12 bar per policy v2.",
            chunk_size=30,
            chunk_overlap=0,
            supersedes_document_uid=v1.document_uid,
        )
        _cleanup.append(v2.document_id)

        old_doc = await session.get(Document, v1.document_id)
        new_doc = await session.get(Document, v2.document_id)

        assert old_doc.status == DocumentStatus.SUPERSEDED
        assert old_doc.version == 1
        assert new_doc.status == DocumentStatus.ACTIVE
        assert new_doc.version == 2
        assert new_doc.supersedes_id == old_doc.id


@pytest.mark.asyncio
async def test_search_prefers_current_version_by_default(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        v1 = await ingest_document(
            session,
            "policy_v1.txt",
            b"Pressure limit UNIQTERM456 is 10 bar per policy v1.",
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(v1.document_id)

        v2 = await ingest_document(
            session,
            "policy_v2.txt",
            b"Pressure limit UNIQTERM456 is 12 bar per policy v2.",
            chunk_size=30,
            chunk_overlap=0,
            supersedes_document_uid=v1.document_uid,
        )
        _cleanup.append(v2.document_id)
        invalidate_bm25_index()

        # mode="vector" here (not "bm25"): with only these 2 documents in
        # the corpus and both sharing the query term equally, BM25's IDF
        # degenerates to ~0 for that term (same tiny-corpus characteristic
        # documented in tests/unit/test_bm25.py) — vector search isn't
        # corpus-size-dependent the same way, so it isolates the
        # versioning-filter behavior this test actually checks.
        default_results = await run_search(
            session, "UNIQTERM456 pressure limit", mode="vector", rerank=False
        )
        assert all(r.document_id != str(v1.document_id) for r in default_results)
        assert any(r.document_id == str(v2.document_id) for r in default_results)

        historical_results = await run_search(
            session,
            "UNIQTERM456 pressure limit",
            mode="vector",
            rerank=False,
            include_historical=True,
        )
        document_ids_seen = {r.document_id for r in historical_results}
        assert str(v1.document_id) in document_ids_seen
        assert str(v2.document_id) in document_ids_seen


@pytest.mark.asyncio
async def test_supersedes_unknown_document_uid_is_rejected(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        with pytest.raises(IngestionError, match="unknown document_uid"):
            await ingest_document(
                session,
                "orphan_v2.txt",
                b"This claims to supersede something that does not exist.",
                supersedes_document_uid="not-a-real-document-uid",
            )
