"""Requires postgres + qdrant running; downloads the real embedding and
CrossEncoder reranker models on first run.
"""

import shutil

import pytest

from app.models.document import Document
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import ingest_document
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
        for document_id in created_document_ids:
            delete_document_vectors(document_id)
            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)
        await session.commit()
    invalidate_bm25_index()

    if RAW_STORAGE_ROOT.exists():
        shutil.rmtree(RAW_STORAGE_ROOT, ignore_errors=True)


async def _ingest(session, filename: str, text: str, ids: list):
    result = await ingest_document(session, filename, text.encode(), chunk_size=30, chunk_overlap=0)
    ids.append(result.document_id)
    return result


@pytest.mark.asyncio
async def test_rerank_populates_rerank_score_when_enabled(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        await _ingest(
            session,
            "confined_space.txt",
            "Confined space entry requires a permit, gas testing, and a standby attendant.",
            _cleanup,
        )
        await _ingest(
            session,
            "fire_drill.txt",
            "The safety procedure for annual fire drill scheduling is documented separately.",
            _cleanup,
        )
        invalidate_bm25_index()

        reranked = await run_search(
            session, "confined space entry safety requirements", top_k=5, rerank=True
        )
        not_reranked = await run_search(
            session, "confined space entry safety requirements", top_k=5, rerank=False
        )

        assert all(r.rerank_score is not None for r in reranked)
        assert all(r.rerank_score is None for r in not_reranked)


@pytest.mark.asyncio
async def test_rerank_surfaces_correct_top_result(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        await _ingest(
            session,
            "confined_space.txt",
            "Confined space entry requires a permit, gas testing, and a standby attendant.",
            _cleanup,
        )
        await _ingest(
            session,
            "fire_drill.txt",
            "The safety procedure for annual fire drill scheduling is documented separately.",
            _cleanup,
        )
        await _ingest(
            session,
            "finance.txt",
            "Quarterly financial reporting is unrelated to plant instrumentation work.",
            _cleanup,
        )
        invalidate_bm25_index()

        results = await run_search(
            session, "confined space entry safety requirements", top_k=3, rerank=True
        )

        assert results
        assert "Confined space entry" in results[0].text
        # rerank_score should determine order (descending)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)
