"""Second-stage reranking with a HuggingFace CrossEncoder.

Why rerank at all (see docs/retrieval-design.md): a bi-encoder (the
embedding model used for vector search) encodes the query and each chunk
independently, so it can never model interactions between specific query
and chunk tokens — it's fast enough to run over the whole corpus but
approximate. A CrossEncoder instead runs the query and a candidate chunk
through the model TOGETHER, attending across both, which is far more
accurate at judging "is this chunk actually relevant to this query" — but
too slow to run over an entire corpus. The standard pattern (used here) is
therefore: cheap retrieval (vector+BM25, hundreds of candidates) narrowed
to a few dozen, then this precise but expensive reranker picks the final
handful of evidence chunks.
"""

import time
from functools import lru_cache

import structlog

from app.config import get_settings

logger = structlog.get_logger("reranking")


class RerankingError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    logger.info("reranker_model_loading", model=model_name)
    model = CrossEncoder(model_name)
    logger.info("reranker_model_loaded", model=model_name)
    return model


class RerankerService:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_retries: int = 2,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size or settings.reranker_batch_size
        self.max_retries = max_retries

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """candidates: [(id, text), ...] in any order. Returns [(id, score), ...]
        sorted best-first, truncated to top_k if given."""
        if not candidates:
            return []

        model = _load_reranker(self.model_name)
        pairs = [(query, text) for _, text in candidates]
        scores = self._predict_with_retry(model, pairs)

        ranked = sorted(
            zip([c[0] for c in candidates], scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k] if top_k is not None else ranked

    def _predict_with_retry(self, model, pairs: list[tuple[str, str]]) -> list[float]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return model.predict(pairs, batch_size=self.batch_size).tolist()
            except Exception as exc:  # transient model/backend hiccup
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 0.5 * (2**attempt)
                    logger.warning(
                        "rerank_batch_retry", attempt=attempt, error=str(exc), wait_seconds=wait
                    )
                    time.sleep(wait)
        raise RerankingError(f"Reranking failed after {self.max_retries + 1} attempts: {last_exc}")


@lru_cache(maxsize=1)
def get_reranker_service() -> RerankerService:
    return RerankerService()
