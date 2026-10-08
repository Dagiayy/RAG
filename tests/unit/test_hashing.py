from app.services.ingestion.hashing import sha256_bytes, sha256_text


def test_sha256_bytes_is_deterministic():
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")


def test_sha256_bytes_differs_for_different_content():
    assert sha256_bytes(b"hello") != sha256_bytes(b"world")


def test_sha256_text_matches_bytes_encoding():
    assert sha256_text("hello") == sha256_bytes(b"hello")
