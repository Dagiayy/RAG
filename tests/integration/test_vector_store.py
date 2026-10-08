"""Requires `docker compose up -d qdrant` running."""

import uuid

import pytest

from app.models.document import AccessLevel, Document, DocumentChunk, DocumentStatus, SourceType
from app.services.retrieval.vector import (
    delete_document_vectors,
    ensure_collection,
    get_qdrant_client,
    search,
    upsert_chunks,
)

TEST_COLLECTION = "test_enterprise_chunks"


def _make_document(**overrides) -> Document:
    defaults = dict(
        id=uuid.uuid4(),
        document_uid=uuid.uuid4().hex,
        title="Test Doc",
        source_filename="test.txt",
        source_type=SourceType.TXT,
        version=1,
        status=DocumentStatus.ACTIVE,
        access_level=AccessLevel.INTERNAL,
        checksum="deadbeef",
        source_uri="data/raw/test.txt",
        department="engineering",
        author=None,
        effective_date=None,
        expiry_date=None,
    )
    defaults.update(overrides)
    return Document(**defaults)


def _make_chunk(document_id: uuid.UUID, text: str, **overrides) -> DocumentChunk:
    defaults = dict(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=0,
        text=text,
        page_number=1,
        section=None,
        paragraph=None,
        checksum="c0ffee",
        embedding_model="fake",
        embedding_model_version="v1",
    )
    defaults.update(overrides)
    return DocumentChunk(**defaults)


@pytest.fixture(autouse=True)
def _cleanup_collection():
    yield
    client = get_qdrant_client()
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


def test_upsert_and_search_returns_matching_chunk():
    document = _make_document()
    chunk_a = _make_chunk(document.id, "the quick brown fox")
    chunk_b = _make_chunk(document.id, "an unrelated sentence about finance")

    ensure_collection(vector_size=3, collection_name=TEST_COLLECTION)
    upsert_chunks(
        document,
        [chunk_a, chunk_b],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        collection_name=TEST_COLLECTION,
    )

    results = search([1.0, 0.0, 0.0], top_k=1, collection_name=TEST_COLLECTION)

    assert len(results) == 1
    assert results[0].chunk_id == str(chunk_a.id)
    assert results[0].payload["text"] == "the quick brown fox"


def test_search_with_access_level_filter():
    document_public = _make_document(access_level=AccessLevel.PUBLIC)
    document_restricted = _make_document(access_level=AccessLevel.RESTRICTED)
    chunk_public = _make_chunk(document_public.id, "public info")
    chunk_restricted = _make_chunk(document_restricted.id, "restricted info")

    ensure_collection(vector_size=2, collection_name=TEST_COLLECTION)
    upsert_chunks(document_public, [chunk_public], [[1.0, 0.0]], collection_name=TEST_COLLECTION)
    upsert_chunks(
        document_restricted, [chunk_restricted], [[1.0, 0.0]], collection_name=TEST_COLLECTION
    )

    results = search(
        [1.0, 0.0],
        top_k=10,
        filters={"access_level": ["public"]},
        collection_name=TEST_COLLECTION,
    )

    assert len(results) == 1
    assert results[0].payload["text"] == "public info"


def test_delete_document_vectors_removes_all_its_chunks():
    document = _make_document()
    chunk = _make_chunk(document.id, "to be deleted")

    ensure_collection(vector_size=2, collection_name=TEST_COLLECTION)
    upsert_chunks(document, [chunk], [[1.0, 0.0]], collection_name=TEST_COLLECTION)
    assert len(search([1.0, 0.0], top_k=10, collection_name=TEST_COLLECTION)) == 1

    delete_document_vectors(document.id, collection_name=TEST_COLLECTION)

    assert len(search([1.0, 0.0], top_k=10, collection_name=TEST_COLLECTION)) == 0


def test_search_on_nonexistent_collection_returns_empty():
    results = search([1.0, 0.0], top_k=10, collection_name="does_not_exist_collection")
    assert results == []
