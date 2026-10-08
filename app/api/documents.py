import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security.audit import log_action
from app.core.security.auth import get_current_user
from app.models.document import AccessLevel, Document, DocumentChunk
from app.models.user import User
from app.repositories.db import get_db_session
from app.schemas.document import DocumentDetail, DocumentIngestResponse, DocumentSummary
from app.services.ingestion.pipeline import IngestionError, ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest(
    file: UploadFile,
    department: str | None = None,
    author: str | None = None,
    access_level: AccessLevel = AccessLevel.INTERNAL,
    supersedes_document_uid: str | None = None,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> DocumentIngestResponse:
    """`supersedes_document_uid`: pass the `document_uid` of an existing
    document this upload replaces (spec section 20 versioning) — the old
    version is marked superseded rather than deleted and is still
    queryable via `/search`'s `include_historical`. Requires authentication
    (establishes WHO uploaded, for the audit log below) — this does not
    yet check whether the caller's clearance justifies the `access_level`
    they're requesting for the new document; that's a separate, more
    nuanced policy decision left for later (see docs/implementation-status.md).
    """
    content = await file.read()
    try:
        result = await ingest_document(
            session,
            filename=file.filename or "unknown",
            content=content,
            department=department,
            author=author,
            access_level=access_level,
            supersedes_document_uid=supersedes_document_uid,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await log_action(
        session,
        current_user,
        action="ingest_document",
        resource_type="document",
        resource_id=str(result.document_id),
        detail=f"filename={file.filename!r}, access_level={access_level.value}",
    )

    return DocumentIngestResponse(
        document_id=result.document_id,
        document_uid=result.document_uid,
        chunk_count=result.chunk_count,
        near_duplicate_count=result.near_duplicate_count,
        conflicting_claim_count=result.conflicting_claim_count,
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents(session: AsyncSession = Depends(get_db_session)) -> list[Document]:
    result = await session.scalars(select(Document).order_by(Document.created_at.desc()))
    return list(result.all())


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, session: AsyncSession = Depends(get_db_session)
) -> DocumentDetail:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunk_count = await session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    )

    return DocumentDetail(
        **DocumentSummary.model_validate(document).model_dump(),
        checksum=document.checksum,
        source_uri=document.source_uri,
        chunk_count=chunk_count or 0,
    )
