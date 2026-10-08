"""Security test: an unauthorized query must never retrieve restricted
content — access control is enforced inside the retrieval query itself,
not by post-hoc filtering or trusting the LLM to withhold it (see
docs/security.md). Requires postgres + qdrant running; uses the real
FastAPI app end-to-end (auth, retrieval, everything), not mocks.
"""

import shutil
import uuid

import httpx
import pytest

from app.main import app
from app.models.document import AccessLevel, Document
from app.models.user import ClearanceLevel, User
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import ingest_document
from app.services.ingestion.storage import RAW_STORAGE_ROOT
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import delete_document_vectors
from tests.conftest import ollama_is_reachable


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


async def _create_user(session, clearance: ClearanceLevel) -> User:
    user = User(username=_unique("testuser"), clearance_level=clearance)
    session.add(user)
    await session.flush()
    return user


@pytest.mark.asyncio
async def test_low_clearance_user_cannot_retrieve_restricted_content(_cleanup):
    secret_phrase = f"ProjectCodeName{uuid.uuid4().hex[:8]}"
    public_phrase = f"PublicAnnouncement{uuid.uuid4().hex[:8]}"

    factory = get_session_factory()
    async with factory() as session:
        restricted_result = await ingest_document(
            session,
            "restricted.txt",
            f"The {secret_phrase} budget is confidential executive information.".encode(),
            access_level=AccessLevel.RESTRICTED,
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup["document_ids"].append(restricted_result.document_id)

        public_result = await ingest_document(
            session,
            "public.txt",
            f"The {public_phrase} is available to everyone in the company.".encode(),
            access_level=AccessLevel.PUBLIC,
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup["document_ids"].append(public_result.document_id)

        low_clearance_user = await _create_user(session, ClearanceLevel.PUBLIC)
        high_clearance_user = await _create_user(session, ClearanceLevel.RESTRICTED)
        _cleanup["user_ids"].extend([low_clearance_user.id, high_clearance_user.id])
        low_token = low_clearance_user.api_key
        high_token = high_clearance_user.api_key
        await session.commit()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Low-clearance user searching for the restricted phrase: must get
        # zero results referencing it anywhere in the response, not just
        # filtered from a "top result".
        response = await client.post(
            "/search",
            json={"query": secret_phrase, "mode": "hybrid", "top_k": 20, "rerank": False},
            headers={"Authorization": f"Bearer {low_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert all(secret_phrase not in r["text"] for r in body["results"])
        assert all(r["document_id"] != str(restricted_result.document_id) for r in body["results"])

        # Same low-clearance user searching for the PUBLIC phrase: must
        # find it (proves the filter isn't just blocking everything).
        response = await client.post(
            "/search",
            json={"query": public_phrase, "mode": "hybrid", "top_k": 20, "rerank": False},
            headers={"Authorization": f"Bearer {low_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert any(public_phrase in r["text"] for r in body["results"])

        # High-clearance user searching for the restricted phrase: must
        # find it (proves the block above was clearance-based, not a bug
        # hiding the document from everyone).
        response = await client.post(
            "/search",
            json={"query": secret_phrase, "mode": "hybrid", "top_k": 20, "rerank": False},
            headers={"Authorization": f"Bearer {high_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert any(secret_phrase in r["text"] for r in body["results"])


@pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")
@pytest.mark.asyncio
async def test_low_clearance_user_generated_answer_never_cites_restricted_content(_cleanup):
    """Phase 9 extension of the retrieval-leakage test above: the
    generated answer + its citations must never reference restricted
    content for an unauthorized user either — not just the raw /search
    results. This follows transitively from (a) /query's document_filters
    excluding restricted content from document_evidence before generation
    even sees it, and (b) the citation-safety invariant that a citation
    can only reference evidence that was actually retrieved — but is
    tested explicitly end-to-end rather than just assumed from the two
    pieces separately.
    """
    secret_phrase = f"ExecutiveBonus{uuid.uuid4().hex[:8]}"

    factory = get_session_factory()
    async with factory() as session:
        restricted_result = await ingest_document(
            session,
            "restricted_bonus.txt",
            f"The {secret_phrase} pool is confidential and restricted to executives.".encode(),
            access_level=AccessLevel.RESTRICTED,
            chunk_size=30,
            chunk_overlap=0,
        )
        _cleanup["document_ids"].append(restricted_result.document_id)

        low_clearance_user = await _create_user(session, ClearanceLevel.PUBLIC)
        _cleanup["user_ids"].append(low_clearance_user.id)
        low_token = low_clearance_user.api_key
        await session.commit()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30.0
    ) as client:
        response = await client.post(
            "/query",
            json={"query": f"What is the {secret_phrase}?"},
            headers={"Authorization": f"Bearer {low_token}"},
        )
        assert response.status_code == 200
        body = response.json()

        assert secret_phrase not in body["answer"]["answer"]
        assert body["answer"]["confidence"] == "insufficient_evidence"
        assert all(
            c.get("document_id") != str(restricted_result.document_id)
            for c in body["answer"]["citations"]
        )
        assert all(secret_phrase not in r["text"] for r in body["document_evidence"])


@pytest.mark.asyncio
async def test_search_without_credentials_is_rejected():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json={"query": "anything"})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_search_with_invalid_token_is_rejected():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/search",
            json={"query": "anything"},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert response.status_code == 401
