import pytest
from pydantic import ValidationError

from app.schemas.search import SearchRequest


def test_search_request_requires_nonempty_query():
    with pytest.raises(ValidationError):
        SearchRequest(query="")


def test_search_request_defaults():
    request = SearchRequest(query="hello")
    assert request.top_k == 10
    assert request.mode == "hybrid"
    assert request.rerank is True
    assert request.department is None


def test_search_request_top_k_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="hello", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="hello", top_k=101)
