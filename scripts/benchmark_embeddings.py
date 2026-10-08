"""Small Phase 3 benchmark: embedding throughput and Qdrant search latency.

Run: python scripts/benchmark_embeddings.py

This does not use a labeled relevance dataset (that's Phase 11's evaluation
benchmark) — it measures raw throughput/latency against synthetic text, so
regressions in embedding batching or Qdrant round-trip time are visible
without needing the full app running.
"""

import statistics
import time
import uuid

from app.models.document import AccessLevel, Document, DocumentChunk, DocumentStatus, SourceType
from app.services.embeddings.service import EmbeddingService
from app.services.retrieval.vector import (
    ensure_collection,
    get_qdrant_client,
    search,
    upsert_chunks,
)

BENCHMARK_COLLECTION = "benchmark_chunks"

SAMPLE_SENTENCES = [
    "Siemens PLCs are widely used in cement plant automation across our projects.",
    "The safety procedure requires confined-space entry permits above 10 bar pressure.",
    "Our electrical engineering team has over five years of SCADA integration experience.",
    "Project Alpha's client is a cement manufacturer in the Middle East region.",
    "Quarterly financial reporting is unrelated to plant instrumentation work.",
    "Employees with PLC and SCADA experience are listed in the skills registry.",
    "The 2024 policy version was superseded by the 2025 revision in March.",
    "Instrumentation certification requires renewal every three years per vendor spec.",
]


def _make_document() -> Document:
    return Document(
        id=uuid.uuid4(),
        document_uid=uuid.uuid4().hex,
        title="Benchmark Doc",
        source_filename="benchmark.txt",
        source_type=SourceType.TXT,
        version=1,
        status=DocumentStatus.ACTIVE,
        access_level=AccessLevel.INTERNAL,
        checksum=uuid.uuid4().hex,
        source_uri="data/raw/benchmark.txt",
        department=None,
        author=None,
        effective_date=None,
        expiry_date=None,
    )


def _make_chunk(document_id: uuid.UUID, text: str, model_name: str) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        chunk_index=0,
        text=text,
        page_number=None,
        section=None,
        paragraph=None,
        checksum=uuid.uuid4().hex,
        embedding_model=model_name,
        embedding_model_version="v1",
    )


def main() -> None:
    n_repeats = 20  # ~160 texts total, enough to see batching effects without being slow
    texts = SAMPLE_SENTENCES * n_repeats

    service = EmbeddingService()
    print(f"Embedding model: {service.model_name} (dim={service.dimension})")

    start = time.perf_counter()
    vectors = service.encode(texts)
    embed_seconds = time.perf_counter() - start
    print(
        f"Embedded {len(texts)} texts in {embed_seconds:.2f}s "
        f"({len(texts) / embed_seconds:.1f} texts/sec, batch_size={service.batch_size})"
    )

    document = _make_document()
    chunks = [_make_chunk(document.id, t, service.model_name) for t in texts]

    ensure_collection(vector_size=service.dimension, collection_name=BENCHMARK_COLLECTION)
    start = time.perf_counter()
    upsert_chunks(document, chunks, vectors, collection_name=BENCHMARK_COLLECTION)
    upsert_seconds = time.perf_counter() - start
    print(f"Upserted {len(chunks)} points to Qdrant in {upsert_seconds:.2f}s")

    query_latencies = []
    for query in ["Siemens PLC experience", "cement plant safety procedure", "financial reporting"]:
        query_vector = service.encode_one(query)
        start = time.perf_counter()
        results = search(query_vector, top_k=5, collection_name=BENCHMARK_COLLECTION)
        query_latencies.append((time.perf_counter() - start) * 1000)
        top_text = results[0].payload["text"][:60]
        print(f"  query={query!r} -> top result score={results[0].score:.3f} ({top_text}...)")

    print(
        f"Qdrant search latency: mean={statistics.mean(query_latencies):.1f}ms "
        f"p95~={max(query_latencies):.1f}ms over {len(query_latencies)} queries"
    )

    get_qdrant_client().delete_collection(BENCHMARK_COLLECTION)


if __name__ == "__main__":
    main()
