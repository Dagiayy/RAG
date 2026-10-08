import pytest

from app.services.reranking import service as service_module
from app.services.reranking.service import RerankerService, RerankingError


class FakeCrossEncoder:
    """Deterministic stand-in: scores a (query, text) pair by how many
    query words appear in the text, so results are predictable without
    downloading/running a real CrossEncoder model."""

    def __init__(self, fail_times: int = 0):
        self.calls: list[list[tuple[str, str]]] = []
        self._fail_times = fail_times

    def predict(self, pairs, batch_size=32):
        import numpy as np

        self.calls.append(list(pairs))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("simulated transient failure")
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return np.array(scores)


@pytest.fixture(autouse=True)
def _clear_model_cache():
    service_module._load_reranker.cache_clear()
    yield
    service_module._load_reranker.cache_clear()


def _install_fake_model(monkeypatch, fake: FakeCrossEncoder):
    monkeypatch.setattr(service_module, "_load_reranker", lambda model_name: fake)


def test_rerank_empty_candidates_returns_empty(monkeypatch):
    fake = FakeCrossEncoder()
    _install_fake_model(monkeypatch, fake)

    svc = RerankerService(model_name="fake-reranker")
    assert svc.rerank("query", []) == []
    assert fake.calls == []


def test_rerank_orders_by_relevance_score(monkeypatch):
    fake = FakeCrossEncoder()
    _install_fake_model(monkeypatch, fake)

    svc = RerankerService(model_name="fake-reranker")
    candidates = [
        ("c1", "completely unrelated financial content"),
        ("c2", "Siemens PLC and SCADA automation experience"),
        ("c3", "cement plant safety procedure"),
    ]
    results = svc.rerank("Siemens PLC automation", candidates)

    assert results[0][0] == "c2"


def test_rerank_respects_top_k(monkeypatch):
    fake = FakeCrossEncoder()
    _install_fake_model(monkeypatch, fake)

    svc = RerankerService(model_name="fake-reranker")
    candidates = [(f"c{i}", f"document {i} about PLC systems") for i in range(10)]
    results = svc.rerank("PLC systems", candidates, top_k=3)

    assert len(results) == 3


def test_rerank_retries_transient_failures(monkeypatch):
    fake = FakeCrossEncoder(fail_times=1)
    _install_fake_model(monkeypatch, fake)
    monkeypatch.setattr(service_module.time, "sleep", lambda _: None)

    svc = RerankerService(model_name="fake-reranker", max_retries=2)
    results = svc.rerank("query", [("c1", "some text")])

    assert len(results) == 1
    assert len(fake.calls) == 2


def test_rerank_raises_after_exhausting_retries(monkeypatch):
    fake = FakeCrossEncoder(fail_times=99)
    _install_fake_model(monkeypatch, fake)
    monkeypatch.setattr(service_module.time, "sleep", lambda _: None)

    svc = RerankerService(model_name="fake-reranker", max_retries=1)
    with pytest.raises(RerankingError):
        svc.rerank("query", [("c1", "some text")])
