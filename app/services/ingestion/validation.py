import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.config import get_settings
from app.models.document import SourceType

EXTENSION_TO_SOURCE_TYPE: dict[str, SourceType] = {
    ".pdf": SourceType.PDF,
    ".docx": SourceType.DOCX,
    ".txt": SourceType.TXT,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".csv": SourceType.CSV,
    ".json": SourceType.JSON,
    ".xlsx": SourceType.XLSX,
}

# Sniffed from file bytes, not trusted from the client-supplied Content-Type.
MAGIC_BYTES: dict[bytes, SourceType] = {
    b"%PDF-": SourceType.PDF,
    b"PK\x03\x04": SourceType.DOCX,  # docx and xlsx are both zip; extension disambiguates
}


class FileValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedUpload:
    safe_filename: str
    original_filename: str
    source_type: SourceType
    size_bytes: int


def _safe_extension(filename: str) -> str:
    """Extracts the extension without trusting path components in `filename`
    (path-traversal protection — see docs/security.md)."""
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or name in (".", ".."):
        raise FileValidationError("Invalid filename.")
    suffix = PurePosixPath(name).suffix.lower()
    return suffix


def validate_upload(filename: str, content: bytes) -> ValidatedUpload:
    settings = get_settings()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) == 0:
        raise FileValidationError("Uploaded file is empty.")
    if len(content) > max_bytes:
        raise FileValidationError(f"File exceeds max upload size of {settings.max_upload_mb}MB.")

    extension = _safe_extension(filename)
    if extension not in EXTENSION_TO_SOURCE_TYPE:
        allowed = ", ".join(sorted(EXTENSION_TO_SOURCE_TYPE))
        raise FileValidationError(f"Unsupported file extension '{extension}'. Allowed: {allowed}")

    source_type = EXTENSION_TO_SOURCE_TYPE[extension]

    # Content-sniff PDF/DOCX/XLSX against declared extension; plain-text
    # formats (txt/md/csv/json) have no reliable magic bytes to check.
    if source_type == SourceType.PDF and not content.startswith(b"%PDF-"):
        raise FileValidationError("File claims to be a PDF but does not have a PDF header.")
    if source_type in (SourceType.DOCX, SourceType.XLSX) and not content.startswith(b"PK\x03\x04"):
        raise FileValidationError(
            f"File claims to be a {source_type.value} but is not a valid zip-based Office file."
        )

    safe_filename = f"{uuid.uuid4()}{extension}"
    return ValidatedUpload(
        safe_filename=safe_filename,
        original_filename=PurePosixPath(filename.replace("\\", "/")).name,
        source_type=source_type,
        size_bytes=len(content),
    )
