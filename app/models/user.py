"""Dev-mode auth model (see docs/security.md "Authentication / Authorization").

Local/dev auth is a static user table with a bearer-token lookup
(`api_key`), behind the `AuthProvider` protocol in
app/core/security/auth.py — production deployment swaps in real OIDC/SSO
behind the same protocol without callers changing (see
docs/deployment.md).
"""

import enum
import secrets
import uuid

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class UserRole(enum.StrEnum):
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"
    VIEWER = "viewer"


class ClearanceLevel(enum.StrEnum):
    """Ordered the same as `AccessLevel` on Document — see
    app/core/security/access.py for the ordering/comparison logic."""

    PUBLIC = "public"
    INTERNAL = "internal"
    DEPARTMENT = "department"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


def _generate_api_key() -> str:
    return secrets.token_urlsafe(32)


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(128), unique=True)
    api_key: Mapped[str] = mapped_column(String(64), unique=True, default=_generate_api_key)
    role: Mapped[UserRole] = mapped_column(String(16), default=UserRole.VIEWER)
    department: Mapped[str | None] = mapped_column(String(128), nullable=True)
    clearance_level: Mapped[ClearanceLevel] = mapped_column(
        String(16), default=ClearanceLevel.PUBLIC
    )
