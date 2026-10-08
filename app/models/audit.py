import enum
import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class IngestionStatus(enum.StrEnum):
    SUCCESS = "success"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class IngestionAuditLog(Base, TimestampMixin):
    """One row per ingestion attempt — see docs pipeline step 'ingestion audit
    record'. Kept separate from the broader RBAC `audit_log` (Phase 8) since
    ingestion happens before a document (or any user-facing resource) exists.
    """

    __tablename__ = "ingestion_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    source_filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[IngestionStatus] = mapped_column(String(16))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base, TimestampMixin):
    """General RBAC audit trail (docs/data-model.md `audit_log` /
    docs/security.md "Audit logging"): every authenticated action against a
    user-facing resource. Never stores raw document/chunk text — only IDs,
    counts, and outcome, matching the "never log sensitive document
    contents" rule in docs/security.md.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
