"""Requires postgres + qdrant running; downloads the real embedding model
on first run. Demonstrates why lexical (BM25) and semantic (vector) search
complement each other (docs/retrieval-design.md): an exact project-code
match that a small embedding model may not surface well is still found via
BM25, and gets fused in.
"""

import shutil

import pytest

from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.hybrid import hybrid_search, run_search
from app.services.retrieval.vector import delete_document_vectors


@pytest.fixture(autouse=True)
async def _cleanup():
    created_document_ids: list = []
    yield created_document_ids

    factory = get_session_factory()
    async with factory() as session:
        from app.models.document import Document

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
async def test_bm25_finds_exact_code_vector_search_may_rank_low(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        # BM25's IDF term goes to ~0 for a word that appears in most of a
        # tiny corpus, so this needs several unrelated documents to give
        # "XJ-4471" a realistic (non-degenerate) document frequency — see
        # the same note in tests/unit/test_bm25.py.
        await _ingest(
            session,
            "project_code.txt",
            "Project code XJ-4471 covers the automation retrofit for the northern facility.",
            _cleanup,
        )
        await _ingest(
            session,
            "unrelated1.txt",
            "The cafeteria menu changes every Tuesday and Friday this quarter.",
            _cleanup,
        )
        await _ingest(
            session,
            "unrelated2.txt",
            "Annual leave requests must be submitted two weeks in advance.",
            _cleanup,
        )
        await _ingest(
            session,
            "unrelated3.txt",
            "The office parking garage will be repaved next month.",
            _cleanup,
        )
        invalidate_bm25_index()

        # rerank=False: this test is about raw BM25 retrieval finding the
        # exact code, not about reranking (see test_reranking.py for that).
        bm25_results = await run_search(session, "XJ-4471", mode="bm25", top_k=5, rerank=False)
        assert any(r.text.find("XJ-4471") != -1 for r in bm25_results)


@pytest.mark.asyncio
async def test_hybrid_search_fuses_vector_and_bm25_results(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        await _ingest(
            session,
            "plc_skills.txt",
            "Our electrical engineers have Siemens PLC and SCADA automation experience.",
            _cleanup,
        )
        await _ingest(
            session,
            "cement_projects.txt",
            "Several projects for cement manufacturing clients used PLC-based control systems.",
            _cleanup,
        )
        await _ingest(
            session,
            "finance.txt",
            "Quarterly financial reporting is unrelated to plant instrumentation work.",
            _cleanup,
        )
        invalidate_bm25_index()

        results = await hybrid_search(session, "Siemens PLC automation experience", top_k=5)

        assert len(results) >= 1
        top = results[0]
        assert "Siemens" in top.text or "PLC" in top.text
        # top hit should have contributions tracked from at least one retriever
        assert top.vector_score is not None or top.bm25_score is not None


@pytest.mark.asyncio
async def test_hybrid_search_respects_access_level_filter(_cleanup):
    factory = get_session_factory()
    async with factory() as session:
        from app.models.document import AccessLevel

        result_public = await ingest_document(
            session,
            "public_doc.txt",
            b"Public safety guidance for cement plant operations.",
            access_level=AccessLevel.PUBLIC,
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(result_public.document_id)

        result_restricted = await ingest_document(
            session,
            "restricted_doc.txt",
            b"Restricted safety guidance for cement plant operations only for executives.",
            access_level=AccessLevel.RESTRICTED,
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup.append(result_restricted.document_id)
        invalidate_bm25_index()

        results = await hybrid_search(
            session,
            "cement plant safety guidance",
            top_k=10,
            filters={"access_level": ["public"]},
        )

        document_ids = {r.document_id for r in results}
        assert str(result_restricted.document_id) not in document_ids
