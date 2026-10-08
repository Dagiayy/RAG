import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.document import AccessLevel, DocumentStatus, SourceType


class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    document_uid: str
    chunk_count: int
    near_duplicate_count: int = 0
    conflicting_claim_count: int = 0


class DocumentSummary(BaseModel):
    id: uuid.UUID
    document_uid: str
    title: str
    source_filename: str
    source_type: SourceType
    version: int
    status: DocumentStatus
    access_level: AccessLevel
    department: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetail(DocumentSummary):
    checksum: str
    source_uri: str
    chunk_count: int
