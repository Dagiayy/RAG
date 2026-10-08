"""Chunking strategies for indexed text.

Trade-offs (see docs/retrieval-design.md for how these feed retrieval):

- **fixed** (word-count windows with overlap): cheapest, most predictable
  chunk size (good for embedding-model token limits), but cuts through
  sentences/ideas arbitrarily — worst for retrieval precision on its own.
  Word count is used as an inexpensive proxy for token count; swap in a real
  tokenizer (e.g. tiktoken) if an embedding model needs exact token budgets.
- **sentence**: keeps sentences intact, packs them up to `chunk_size` words
  with `overlap` words repeated between windows. Better semantic coherence
  than fixed, still format-agnostic.
- **paragraph**: one chunk per paragraph (merging tiny paragraphs and
  splitting oversized ones against `chunk_size`). Best when the source
  document's own structure is meaningful (policies, procedures).
- **heading_aware**: splits on Markdown/plain-text headings first, then
  applies paragraph packing within each section, and carries the heading
  text into `section` chunk metadata. Best for structured documents (specs,
  manuals) where "what section is this evidence from" matters for citations.
- **semantic** (NOT implemented yet): would merge sentences by embedding
  similarity rather than fixed size. Deferred to Phase 3, once the embedding
  service exists — implementing it now would mean faking similarity
  comparisons, which we don't do (see docs/implementation-status.md).
"""

import enum
import re
from dataclasses import dataclass

from app.services.ingestion.extraction import ExtractedDocument

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 \-:]{3,80})$")


class ChunkingStrategy(enum.StrEnum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HEADING_AWARE = "heading_aware"


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: int | None
    section: str | None
    paragraph: int | None


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack_by_words(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack whole units (sentences or paragraphs) into windows of
    up to `chunk_size` words, repeating the last `overlap` words of context
    at the start of the next window."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        unit_words = len(unit.split())
        if current and current_words + unit_words > chunk_size:
            chunks.append(" ".join(current))
            if overlap > 0:
                tail_words = " ".join(current).split()[-overlap:]
                current = [" ".join(tail_words)] if tail_words else []
                current_words = len(tail_words)
            else:
                current = []
                current_words = 0
        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _chunk_heading_sections(text: str) -> list[tuple[str | None, str]]:
    """Returns (heading, section_text) pairs. Text before the first heading
    gets heading=None."""
    lines = text.split("\n")
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in lines:
        if _HEADING_RE.match(line.strip()):
            heading = line.strip().lstrip("#").strip()
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def chunk_extracted_document(
    extracted: ExtractedDocument,
    strategy: ChunkingStrategy,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """chunk_size/overlap are in words (see module docstring on the 'fixed'
    strategy's word-as-token-proxy trade-off)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[Chunk] = []
    index = 0

    for page in extracted.pages:
        if not page.text.strip():
            continue

        if strategy == ChunkingStrategy.FIXED:
            for text in _chunk_fixed(page.text, chunk_size, overlap):
                chunks.append(Chunk(text, index, page.page_number, None, None))
                index += 1

        elif strategy == ChunkingStrategy.SENTENCE:
            sentences = _split_sentences(page.text)
            for text in _pack_by_words(sentences, chunk_size, overlap):
                chunks.append(Chunk(text, index, page.page_number, None, None))
                index += 1

        elif strategy == ChunkingStrategy.PARAGRAPH:
            paragraphs = _split_paragraphs(page.text)
            for para_num, text in enumerate(
                _pack_by_words(paragraphs, chunk_size, overlap), start=1
            ):
                chunks.append(Chunk(text, index, page.page_number, None, para_num))
                index += 1

        elif strategy == ChunkingStrategy.HEADING_AWARE:
            for heading, section_text in _chunk_heading_sections(page.text):
                paragraphs = _split_paragraphs(section_text)
                for text in _pack_by_words(paragraphs, chunk_size, overlap):
                    chunks.append(Chunk(text, index, page.page_number, heading, None))
                    index += 1
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunks
