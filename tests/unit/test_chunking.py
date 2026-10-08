import pytest

from app.services.ingestion.chunking import ChunkingStrategy, chunk_extracted_document
from app.services.ingestion.extraction import ExtractedDocument, ExtractedPage


def _doc(text: str, page_number: int | None = 1) -> ExtractedDocument:
    return ExtractedDocument(pages=[ExtractedPage(page_number=page_number, text=text)])


def test_fixed_chunking_respects_chunk_size_and_overlap():
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_extracted_document(_doc(text), ChunkingStrategy.FIXED, chunk_size=20, overlap=5)

    # windows of 20 with step 15 over 50 words: starts at 0, 15, 30 (30+20>=50 stops there)
    assert len(chunks) == 3
    assert chunks[0].text.split() == words[0:20]
    assert chunks[-1].text.split() == words[30:50]
    # overlap: last 5 words of chunk 0 should be the first 5 words of chunk 1
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


def test_fixed_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_extracted_document(_doc("a b c"), ChunkingStrategy.FIXED, chunk_size=10, overlap=10)


def test_sentence_chunking_keeps_sentences_intact():
    text = "First sentence here. Second sentence follows. Third one too."
    chunks = chunk_extracted_document(
        _doc(text), ChunkingStrategy.SENTENCE, chunk_size=6, overlap=0
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.text.strip().endswith((".", "!", "?"))


def test_paragraph_chunking_splits_on_blank_lines():
    text = "Paragraph one has some words.\n\nParagraph two has other words.\n\nParagraph three."
    chunks = chunk_extracted_document(
        _doc(text), ChunkingStrategy.PARAGRAPH, chunk_size=3, overlap=0
    )

    assert len(chunks) == 3
    assert all(c.paragraph is not None for c in chunks)


def test_heading_aware_chunking_tags_section():
    text = "# Introduction\nSome intro text here.\n\n# Details\nMore detailed content follows."
    chunks = chunk_extracted_document(
        _doc(text), ChunkingStrategy.HEADING_AWARE, chunk_size=50, overlap=0
    )

    sections = {c.section for c in chunks}
    assert "Introduction" in sections
    assert "Details" in sections


def test_chunks_carry_page_number_from_source_page():
    chunks = chunk_extracted_document(_doc("some content", page_number=7), ChunkingStrategy.FIXED)
    assert all(c.page_number == 7 for c in chunks)


def test_empty_document_produces_no_chunks():
    chunks = chunk_extracted_document(_doc("   "), ChunkingStrategy.PARAGRAPH)
    assert chunks == []
