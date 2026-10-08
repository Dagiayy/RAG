from pathlib import Path

RAW_STORAGE_ROOT = Path("data/raw")


def save_raw_upload(document_uid: str, safe_filename: str, content: bytes) -> str:
    """Persists the original file bytes under data/raw/{document_uid}/{safe_filename}
    (never under a client-controlled path — see docs/security.md path-traversal note).
    Returns the relative source_uri to store on the Document row.
    """
    directory = RAW_STORAGE_ROOT / document_uid
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / safe_filename
    path.write_bytes(content)
    return str(path.as_posix())
