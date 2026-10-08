import pytest

from app.models.document import SourceType
from app.services.ingestion.validation import FileValidationError, validate_upload


def test_accepts_valid_txt_upload():
    result = validate_upload("notes.txt", b"hello world")
    assert result.source_type == SourceType.TXT
    assert result.original_filename == "notes.txt"
    assert result.safe_filename.endswith(".txt")


def test_rejects_empty_file():
    with pytest.raises(FileValidationError):
        validate_upload("empty.txt", b"")


def test_rejects_unsupported_extension():
    with pytest.raises(FileValidationError):
        validate_upload("archive.zip", b"PK\x03\x04rest")


def test_rejects_pdf_without_pdf_header():
    with pytest.raises(FileValidationError):
        validate_upload("fake.pdf", b"not really a pdf")


def test_accepts_pdf_with_valid_header():
    result = validate_upload("real.pdf", b"%PDF-1.4\n...")
    assert result.source_type == SourceType.PDF


def test_path_traversal_in_filename_is_neutralized():
    result = validate_upload("../../etc/passwd.txt", b"content")
    assert "/" not in result.safe_filename
    assert ".." not in result.safe_filename
    assert result.original_filename == "passwd.txt"


def test_oversized_file_rejected(monkeypatch):
    from app.config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    settings_module.get_settings.cache_clear()
    try:
        with pytest.raises(FileValidationError):
            validate_upload("big.txt", b"x" * 1024)
    finally:
        monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
        settings_module.get_settings.cache_clear()
