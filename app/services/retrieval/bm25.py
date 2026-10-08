"""BM25 lexical retrieval (see docs/retrieval-design.md).

Vector search is good at paraphrase/semantic similarity but often misses
exact identifiers — employee IDs, certification names, model numbers,
project codes, acronyms — because embedding models compress rare tokens
into the same neighborhood as common ones. BM25 scores exact term overlap
directly, so it complements vector search rather than competing with it;
see hybrid.py for how the two are fused.

rank-bm25's BM25Okapi is built from a full in-memory corpus (no
incremental-update API), so this module keeps one process-wide index,
rebuilt lazily from PostgreSQL the next time it's queried after being
invalidated (see `invalidate_bm25_index()`, called by the ingestion
pipeline after every successful ingest). Rebuilding on every query would be
correct but wasteful; rebuilding eagerly on every ingest would slow down
ingestion for no benefit if nobody searches in between — lazy rebuild on
next use is the middle ground.
"""

import asyncio
import re
import threading
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/alphanumeric tokenizer. Deliberately simple (no
    stemming/lemmatization) so exact identifiers like 'XJ-4471' or 'SCADA'
    match literally rather than being normalized away."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25ChunkMetadata:
    document_id: str
    text: str
    source_filename: str
    page_number: int | None
    section: str | None
    access_level: str
    department: str | None
    document_status: str


@dataclass
class BM25Index:
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, BM25ChunkMetadata] = field(default_factory=dict)
    _bm25: BM25Okapi | None = None

    @property
    def is_empty(self) -> bool:
        return not self.chunk_ids

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.is_empty or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(chunk_id, score) for chunk_id, score in ranked[:top_k] if score > 0]


def build_bm25_index(rows: list[tuple[str, BM25ChunkMetadata]]) -> BM25Index:
    """rows: list of (chunk_id, metadata) — metadata.text is what's indexed."""
    index = BM25Index()
    if not rows:
        return index

    index.chunk_ids = [chunk_id for chunk_id, _ in rows]
    index.metadata = {chunk_id: meta for chunk_id, meta in rows}
    corpus_tokens = [tokenize(meta.text) for _, meta in rows]
    index._bm25 = BM25Okapi(corpus_tokens)
    return index


_index: BM25Index | None = None
_lock = threading.Lock()


def invalidate_bm25_index() -> None:
    global _index
    with _lock:
        _index = None


def set_bm25_index(index: BM25Index) -> None:
    """Installs a freshly built index (called by the async loader in
    hybrid.py after fetching chunk rows from Postgres, since building the
    index itself is sync/CPU work)."""
    global _index
    with _lock:
        _index = index


def get_current_bm25_index() -> BM25Index | None:
    """Returns the cached index if one is loaded, or None if it needs
    (re)building — callers use this to decide whether to fetch fresh data."""
    with _lock:
        return _index


async def ensure_bm25_index(session: AsyncSession) -> BM25Index:
    """Returns the cached index, rebuilding it from PostgreSQL first if it
    was invalidated (or never built). The Postgres fetch is async; the
    tokenization/BM25Okapi construction is CPU-bound and runs in a thread.
    """
    cached = get_current_bm25_index()
    if cached is not None:
        return cached

    result = await session.execute(
        select(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
    )
    rows = [
        (
            str(chunk.id),
            BM25ChunkMetadata(
                document_id=str(document.id),
                text=chunk.text,
                source_filename=document.source_filename,
                page_number=chunk.page_number,
                section=chunk.section,
                access_level=document.access_level,
                department=document.department,
                document_status=document.status,
            ),
        )
        for chunk, document in result.all()
    ]

    index = await asyncio.to_thread(build_bm25_index, rows)
    set_bm25_index(index)
    return index
