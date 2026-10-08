"""Qdrant vector store integration: collection management, indexing, search.

Payload fields mirror docs/data-model.md's "chunk metadata" list so
retrieval-time filtering (access control, effective dates, department —
see docs/security.md) can happen inside the Qdrant query itself, before any
content reaches the LLM.
"""

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.models.document import Document, DocumentChunk

logger = structlog.get_logger("vector_store")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(vector_size: int, collection_name: str | None = None) -> str:
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        logger.info("qdrant_collection_created", collection=name, vector_size=vector_size)
    return name


def _to_payload(document: Document, chunk: DocumentChunk) -> dict:
    def _iso(value: date | datetime | None) -> str | None:
        return value.isoformat() if value else None

    return {
        "document_id": str(document.id),
        "document_uid": document.document_uid,
        "document_version": document.version,
        "source_filename": document.source_filename,
        "source_type": document.source_type,
        "chunk_id": str(chunk.id),
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "paragraph": chunk.paragraph,
        "department": document.department,
        "author": document.author,
        "created_at": _iso(document.created_at),
        "effective_date": _iso(document.effective_date),
        "expiry_date": _iso(document.expiry_date),
        "access_level": document.access_level,
        "document_status": document.status,
        "source_uri": document.source_uri,
        "checksum": chunk.checksum,
        "embedding_model": chunk.embedding_model,
        "embedding_model_version": chunk.embedding_model_version,
        "text": chunk.text,
    }


def upsert_chunks(
    document: Document,
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    collection_name: str | None = None,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")
    if not chunks:
        return

    collection_name = ensure_collection(
        vector_size=len(vectors[0]), collection_name=collection_name
    )
    client = get_qdrant_client()

    points = [
        qmodels.PointStruct(id=str(chunk.id), vector=vector, payload=_to_payload(document, chunk))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)
    logger.info(
        "qdrant_chunks_indexed",
        document_id=str(document.id),
        chunk_count=len(points),
        collection=collection_name,
    )


def delete_document_vectors(document_id: uuid.UUID, collection_name: str | None = None) -> None:
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    if not client.collection_exists(name):
        return
    client.delete(
        collection_name=name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id", match=qmodels.MatchValue(value=str(document_id))
                    )
                ]
            )
        ),
    )


def update_document_payload(
    document_id: uuid.UUID, payload: dict, collection_name: str | None = None
) -> None:
    """Patches a payload field (e.g. `document_status`) across all of a
    document's existing points without re-embedding/re-upserting vectors.
    Needed because Qdrant's payload is a snapshot taken at ingest time, not
    live-synced with PostgreSQL — a Postgres-side update (e.g. marking a
    document SUPERSEDED on versioning, spec section 20) must be mirrored
    here explicitly or retrieval-time filtering on stale payload data would
    silently keep serving outdated status.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()
    if not client.collection_exists(name):
        return
    client.set_payload(
        collection_name=name,
        payload=payload,
        points=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id", match=qmodels.MatchValue(value=str(document_id))
                )
            ]
        ),
    )


@dataclass
class VectorSearchResult:
    chunk_id: str
    document_id: str
    score: float
    payload: dict


def _build_filter(filters: dict[str, str | int | list] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=value)))
        else:
            conditions.append(
                qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
            )
    return qmodels.Filter(must=conditions)


def search(
    query_vector: list[float],
    top_k: int = 10,
    filters: dict[str, str | int | list] | None = None,
    score_threshold: float | None = None,
    collection_name: str | None = None,
) -> list[VectorSearchResult]:
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    client = get_qdrant_client()

    if not client.collection_exists(name):
        return []

    results = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        query_filter=_build_filter(filters),
        score_threshold=score_threshold,
        with_payload=True,
    ).points

    return [
        VectorSearchResult(
            chunk_id=point.payload["chunk_id"],
            document_id=point.payload["document_id"],
            score=point.score,
            payload=point.payload,
        )
        for point in results
    ]
