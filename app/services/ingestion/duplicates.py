"""Near-duplicate detection (spec section 18): exact duplicates are already
rejected at the document level via `documents.checksum` (see pipeline.py).
This catches content that's *substantially* the same without being
byte-identical — a chunk re-copied with minor edits, a policy paragraph
repeated across documents, a renamed/re-saved file. Detection only: never
blocks ingestion or silently discards anything (source provenance is
preserved per spec — the new document is still indexed normally); findings
are recorded so a human/downstream process can decide what to do, and so
corroborating evidence can eventually be recognized as such rather than
just noise.
"""

import asyncio
import uuid
from dataclasses import dataclass

from app.services.retrieval.vector import search as vector_search


@dataclass
class NearDuplicateMatch:
    new_chunk_id: str
    existing_chunk_id: str
    existing_document_id: str
    similarity: float


async def find_near_duplicate_chunks(
    new_chunk_vectors: list[tuple[str, list[float]]],
    exclude_document_id: uuid.UUID,
    threshold: float = 0.92,
) -> list[NearDuplicateMatch]:
    """For each (chunk_id, vector) in the newly-ingested document, searches
    the rest of the corpus for chunks above `threshold` cosine similarity.
    Excludes the document being ingested (comparing it to itself isn't a
    duplicate finding — it's the same document).
    """
    matches: list[NearDuplicateMatch] = []
    exclude_str = str(exclude_document_id)

    for chunk_id, vector in new_chunk_vectors:
        results = await asyncio.to_thread(vector_search, vector, 5, None, threshold)
        for result in results:
            if result.document_id == exclude_str:
                continue
            matches.append(
                NearDuplicateMatch(
                    new_chunk_id=chunk_id,
                    existing_chunk_id=result.chunk_id,
                    existing_document_id=result.document_id,
                    similarity=result.score,
                )
            )

    return matches
