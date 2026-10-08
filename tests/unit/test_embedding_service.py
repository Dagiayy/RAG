import numpy as np
import pytest

from app.services.embeddings import service as service_module
from app.services.embeddings.service import EmbeddingError, EmbeddingService


class FakeModel:
    """Deterministic stand-in for a sentence-transformers model: returns a
    2D vector encoding (len(text), text.count(' ')) so it's cheap, fast,
    and predictable without downloading/running a real model."""

    def __init__(self, dim: int = 2, fail_times: int = 0):
        self._dim = dim
        self.calls: list[list[str]] = []
        self._fail_times = fail_times

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.calls.append(list(texts))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("simulated transient failure")
        vectors = np.array([[float(len(t)), float(t.count(" "))] for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        return vectors


@pytest.fixture(autouse=True)
def _clear_model_cache():
    service_module._load_model.cache_clear()
    yield
    service_module._load_model.cache_clear()


def _install_fake_model(monkeypatch, fake: FakeModel):
    monkeypatch.setattr(service_module, "_load_model", lambda model_name: fake)


def test_encode_batches_and_returns_one_vector_per_text(monkeypatch):
    fake = FakeModel()
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model", batch_size=2, cache_size=10)
    vectors = svc.encode(["a", "bb", "ccc", "dddd", "eeeee"])

    assert len(vectors) == 5
    # batch_size=2 over 5 uncached texts -> 3 model.encode calls
    assert len(fake.calls) == 3


def test_encode_vectors_are_l2_normalized(monkeypatch):
    fake = FakeModel()
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model")
    [vector] = svc.encode(["hello world"])

    norm = np.linalg.norm(vector)
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_encode_uses_cache_for_repeated_text(monkeypatch):
    fake = FakeModel()
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model", cache_size=10)
    first = svc.encode(["repeat me"])
    second = svc.encode(["repeat me"])

    assert first == second
    assert len(fake.calls) == 1  # second call was a pure cache hit


def test_encode_empty_list_returns_empty_without_calling_model(monkeypatch):
    fake = FakeModel()
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model")
    assert svc.encode([]) == []
    assert fake.calls == []


def test_encode_retries_transient_failures(monkeypatch):
    fake = FakeModel(fail_times=1)
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model", max_retries=2)
    monkeypatch.setattr(service_module.time, "sleep", lambda _: None)

    vectors = svc.encode(["retry me"])
    assert len(vectors) == 1
    assert len(fake.calls) == 2  # one failure + one success


def test_encode_raises_embedding_error_after_exhausting_retries(monkeypatch):
    fake = FakeModel(fail_times=99)
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model", max_retries=1)
    monkeypatch.setattr(service_module.time, "sleep", lambda _: None)

    with pytest.raises(EmbeddingError):
        svc.encode(["always fails"])


def test_cache_evicts_least_recently_used(monkeypatch):
    fake = FakeModel()
    _install_fake_model(monkeypatch, fake)

    svc = EmbeddingService(model_name="fake-model", cache_size=2)
    svc.encode(["a"])
    svc.encode(["b"])
    svc.encode(["c"])  # evicts "a" (least recently used)

    calls_before = len(fake.calls)
    svc.encode(["a"])  # cache miss again -> new call
    assert len(fake.calls) == calls_before + 1
