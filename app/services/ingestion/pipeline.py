"""Document ingestion pipeline (Phase 2 + Phase 3 + Phase 4 slice).

Covers: file validation -> type detection -> extraction -> chunking ->
PostgreSQL persistence (document registry + chunks) -> embedding -> Qdrant
indexing -> BM25 index invalidation -> ingestion audit record.
Knowledge-graph entity extraction (Phase 6/7) is not wired in yet — see
docs/implementation-status.md for what's wired up so far.

Exact duplicate detection is enforced here via the `documents.checksum`
unique constraint: re-ingesting byte-identical content is rejected, not
silently re-processed, and the attempt is still recorded in the audit log
(see docs/architecture.md section 18 "duplicate document handling").

Embedding and Qdrant upsert are CPU/IO-bound synchronous calls (the
sentence-transformers model and the qdrant-client library are both sync),
so they run via `asyncio.to_thread` to avoid blocking the event loop.
"""

import asyncio
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import IngestionAuditLog, IngestionStatus
from app.models.document import AccessLevel, Document, DocumentChunk, DocumentStatus
from app.services.embeddings.service import EmbeddingService, get_embedding_service
from app.services.ingestion.chunking import ChunkingStrategy, chunk_extracted_document
from app.services.ingestion.conflict_detection import find_conflicting_claims
from app.services.ingestion.duplicates import find_near_duplicate_chunks
from app.services.ingestion.extraction import ExtractionError, extract
from app.services.ingestion.hashing import sha256_bytes, sha256_text
from app.services.ingestion.storage import save_raw_upload
from app.services.ingestion.validation import FileValidationError, validate_upload
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import update_document_payload, upsert_chunks

logger = structlog.get_logger("ingestion")


class IngestionError(ValueError):
    pass


@dataclass
class IngestionResult:
    document_id: uuid.UUID
    document_uid: str
    chunk_count: int
    near_duplicate_count: int = 0
    conflicting_claim_count: int = 0


async def ingest_document(
    session: AsyncSession,
    filename: str,
    content: bytes,
    *,
    title: str | None = None,
    department: str | None = None,
    author: str | None = None,
    access_level: AccessLevel = AccessLevel.INTERNAL,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    embedding_service: EmbeddingService | None = None,
    index_vectors: bool = True,
    supersedes_document_uid: str | None = None,
) -> IngestionResult:
    """`supersedes_document_uid`: when given, this upload is a new version
    of an existing document lineage (spec section 20 "Document
    Versioning"). The previous version is looked up by `document_uid`, its
    `status` is set to SUPERSEDED, and the new row gets
    `version = previous.version + 1` and `supersedes_id` pointing at it.
    Retrieval prefers non-superseded ("active") documents by default (see
    app/services/retrieval/hybrid.py's `include_historical` parameter) but
    historical versions remain queryable explicitly — they are never
    deleted.
    """
    try:
        validated = validate_upload(filename, content)
    except FileValidationError as exc:
        await _record_audit(session, None, filename, IngestionStatus.FAILED, str(exc))
        raise IngestionError(str(exc)) from exc

    checksum = sha256_bytes(content)

    existing = await session.scalar(select(Document).where(Document.checksum == checksum))
    if existing is not None:
        await _record_audit(
            session,
            existing.id,
            filename,
            IngestionStatus.DUPLICATE,
            f"Identical content already ingested as document {existing.document_uid}.",
        )
        raise IngestionError(
            f"Duplicate content: identical to existing document {existing.document_uid}."
        )

    previous_version: Document | None = None
    if supersedes_document_uid is not None:
        previous_version = await session.scalar(
            select(Document).where(Document.document_uid == supersedes_document_uid)
        )
        if previous_version is None:
            await _record_audit(
                session,
                None,
                filename,
                IngestionStatus.FAILED,
                f"supersedes_document_uid {supersedes_document_uid!r} not found.",
            )
            raise IngestionError(
                f"Cannot supersede unknown document_uid {supersedes_document_uid!r}."
            )

    try:
        extracted = extract(validated.source_type, content)
    except ExtractionError as exc:
        await _record_audit(session, None, filename, IngestionStatus.FAILED, str(exc))
        raise IngestionError(str(exc)) from exc

    chunks = chunk_extracted_document(extracted, chunking_strategy, chunk_size, chunk_overlap)
    if not chunks:
        await _record_audit(
            session, None, filename, IngestionStatus.FAILED, "No content extracted to chunk."
        )
        raise IngestionError("Document produced no chunkable content.")

    document_uid = uuid.uuid4().hex
    source_uri = save_raw_upload(document_uid, validated.safe_filename, content)

    document = Document(
        document_uid=document_uid,
        title=title or validated.original_filename,
        source_filename=validated.original_filename,
        source_type=validated.source_type,
        checksum=checksum,
        source_uri=source_uri,
        department=department,
        author=author,
        access_level=access_level,
        version=(previous_version.version + 1) if previous_version else 1,
        supersedes_id=previous_version.id if previous_version else None,
    )
    if previous_version is not None:
        previous_version.status = DocumentStatus.SUPERSEDED
    session.add(document)
    await session.flush()  # assigns document.id

    chunk_rows = [
        DocumentChunk(
            document_id=document.id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            section=chunk.section,
            paragraph=chunk.paragraph,
            checksum=sha256_text(chunk.text),
        )
        for chunk in chunks
    ]
    session.add_all(chunk_rows)
    await session.flush()  # assigns chunk_rows[*].id, needed as Qdrant point ids

    near_duplicate_count = 0
    conflicting_claim_count = 0
    if index_vectors:
        service = embedding_service or get_embedding_service()
        vectors = await asyncio.to_thread(service.encode, [row.text for row in chunk_rows])
        for row in chunk_rows:
            row.embedding_model = service.model_name
            row.embedding_model_version = service.model_version

        # Search BEFORE upserting: the new document's own vectors aren't in
        # Qdrant yet, so there's no self-match risk to filter out.
        near_duplicates = await find_near_duplicate_chunks(
            [(str(row.id), vector) for row, vector in zip(chunk_rows, vectors, strict=True)],
            exclude_document_id=document.id,
        )
        near_duplicate_count = len(near_duplicates)
        if near_duplicates:
            logger.info(
                "near_duplicates_detected",
                document_id=str(document.id),
                match_count=near_duplicate_count,
                other_document_ids=sorted({m.existing_document_id for m in near_duplicates}),
            )

        # Also searched BEFORE upserting, same self-match-avoidance reason
        # as near-duplicate detection above.
        conflicts = await find_conflicting_claims(
            [(str(row.id), vector) for row, vector in zip(chunk_rows, vectors, strict=True)],
            {str(row.id): row.text for row in chunk_rows},
            exclude_document_id=document.id,
        )
        conflicting_claim_count = len(conflicts)
        if conflicts:
            logger.info(
                "conflicting_claims_detected",
                document_id=str(document.id),
                match_count=conflicting_claim_count,
                other_document_ids=sorted({c.existing_document_id for c in conflicts}),
            )

        await asyncio.to_thread(upsert_chunks, document, chunk_rows, vectors)

        if previous_version is not None:
            await asyncio.to_thread(
                update_document_payload,
                previous_version.id,
                {"document_status": DocumentStatus.SUPERSEDED.value},
            )

    audit_parts = []
    if near_duplicate_count:
        audit_parts.append(f"near_duplicate_chunk_matches={near_duplicate_count}")
    if conflicting_claim_count:
        audit_parts.append(f"conflicting_claim_matches={conflicting_claim_count}")
    audit_detail = "; ".join(audit_parts) or None
    await _record_audit(session, document.id, filename, IngestionStatus.SUCCESS, audit_detail)
    await session.commit()
    invalidate_bm25_index()

    logger.info(
        "document_ingested",
        document_id=str(document.id),
        document_uid=document.document_uid,
        source_type=validated.source_type.value,
        chunk_count=len(chunks),
    )

    return IngestionResult(
        document_id=document.id,
        document_uid=document.document_uid,
        chunk_count=len(chunks),
        near_duplicate_count=near_duplicate_count,
        conflicting_claim_count=conflicting_claim_count,
    )


async def _record_audit(
    session: AsyncSession,
    document_id: uuid.UUID | None,
    filename: str,
    status: IngestionStatus,
    detail: str | None,
) -> None:
    session.add(
        IngestionAuditLog(
            document_id=document_id,
            source_filename=filename,
            status=status,
            detail=detail,
        )
    )
    await session.commit()
