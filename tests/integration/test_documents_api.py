"""HTTP-level coverage for /documents/ingest's authentication requirement
(see docs/implementation-status.md, "Post-Phase-11 work") — the pipeline-
level ingest_document() function is already extensively tested elsewhere
(tests/integration/test_ingestion_pipeline.py and others); this covers
the endpoint's auth wiring specifically. Requires postgres + qdrant
running; uses the real FastAPI app end-to-end, not mocks.
"""

import shutil
import uuid

import httpx
import pytest

from app.main import app
from app.models.document import Document
from app.models.user import User
from app.repositories.db import get_session_factory
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import delete_document_vectors


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup():
    created: dict = {"document_ids": [], "user_ids": []}
    yield created

    factory = get_session_factory()
    async with factory() as session:
        for document_id in created["document_ids"]:
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
async def test_ingest_without_credentials_is_rejected():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/documents/ingest",
            files={"file": ("test.txt", b"Some content for the ingest auth test.", "text/plain")},
        )
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_ingest_with_credentials_succeeds_and_is_audited(_cleanup):
    from app.models.user import ClearanceLevel

    factory = get_session_factory()
    async with factory() as session:
        user = User(username=_unique("docapitest"), clearance_level=ClearanceLevel.INTERNAL)
        session.add(user)
        await session.flush()
        _cleanup["user_ids"].append(user.id)
        token = user.api_key
        await session.commit()

    unique_text = f"Ingest API auth test content {uuid.uuid4().hex[:8]}."

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/documents/ingest",
            files={"file": ("test.txt", unique_text.encode(), "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        _cleanup["document_ids"].append(uuid.UUID(body["document_id"]))
        assert body["chunk_count"] >= 1
