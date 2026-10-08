"""Embedding service: text -> normalized vectors.

Design notes (see docs/architecture.md section 8):
- **configurable model**: via `Settings.embedding_model`, not hard-coded.
- **batching**: `encode()` chunks the input list into `batch_size` windows
  before calling the underlying model, so callers can pass arbitrarily
  large chunk lists without controlling batch size themselves.
- **normalization**: vectors are L2-normalized so cosine similarity in
  Qdrant reduces to a dot product (Qdrant's `Cosine` distance already does
  this internally, but normalizing here keeps the vectors well-defined if
  ever consumed outside Qdrant, e.g. for cache comparisons).
- **caching**: an in-memory LRU cache keyed by (model_name, content hash)
  avoids re-embedding identical chunk text (common with duplicate/near-
  duplicate content across documents). Bounded by
  `Settings.embedding_cache_size` — this is a process-local cache, not
  shared across workers; a production deployment would move this to Redis
  (see docs/deployment.md).
- **retry handling**: transient model/backend errors get a small bounded
  retry with backoff rather than failing the whole ingestion batch on one
  hiccup.
- **deterministic IDs / model tracking**: this module doesn't assign chunk
  IDs (that's `DocumentChunk.id`, already deterministic per row); it does
  report `model_name`/`model_version` so callers can stamp vectors with
  what produced them (see `DocumentChunk.embedding_model[_version]`) and
  detect stale embeddings after a model change.
"""

import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import structlog

from app.config import get_settings
from app.services.ingestion.hashing import sha256_text

logger = structlog.get_logger("embeddings")


class EmbeddingError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Process-wide model cache — loading a sentence-transformers model is
    expensive (seconds, plus a one-time download), so it's loaded once per
    process regardless of how many EmbeddingService instances are created."""
    from sentence_transformers import SentenceTransformer

    logger.info("embedding_model_loading", model=model_name)
    model = SentenceTransformer(model_name)
    logger.info("embedding_model_loaded", model=model_name, dim=model.get_embedding_dimension())
    return model


class _LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data: OrderedDict[str, list[float]] = OrderedDict()

    def get(self, key: str) -> list[float] | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: list[float]) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)


class EmbeddingService:
    def __init__(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
        batch_size: int | None = None,
        cache_size: int | None = None,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.model_version = model_version or settings.embedding_model_version
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_retries = max_retries
        self._cache = _LRUCache(cache_size or settings.embedding_cache_size)

    @property
    def dimension(self) -> int:
        return _load_model(self.model_name).get_embedding_dimension()

    def _cache_key(self, text: str) -> str:
        return f"{self.model_name}:{sha256_text(text)}"

    def _encode_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        model = _load_model(self.model_name)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vectors).tolist()
            except Exception as exc:  # model/backend hiccup — retry a bounded number of times
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 0.5 * (2**attempt)
                    logger.warning(
                        "embedding_batch_retry", attempt=attempt, error=str(exc), wait_seconds=wait
                    )
                    time.sleep(wait)
        raise EmbeddingError(f"Embedding failed after {self.max_retries + 1} attempts: {last_exc}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Returns one normalized vector per input text, in the same order.
        Cache hits skip the model entirely; misses are batched together."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(self._cache_key(text))
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append((i, text))

        for batch_start in range(0, len(to_compute), self.batch_size):
            batch = to_compute[batch_start : batch_start + self.batch_size]
            vectors = self._encode_batch_with_retry([t for _, t in batch])
            for (i, text), vector in zip(batch, vectors, strict=True):
                results[i] = vector
                self._cache.put(self._cache_key(text), vector)

        return results  # type: ignore[return-value]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Process-wide default instance so the LRU text cache is actually
    shared across requests instead of being recreated (and emptied) per
    call. Pass an explicit EmbeddingService to bypass this where needed
    (e.g. tests, or a one-off different model)."""
    return EmbeddingService()
