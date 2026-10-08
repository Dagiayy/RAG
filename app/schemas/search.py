from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool = True
    department: str | None = None
    score_threshold: float | None = None
    include_historical: bool = False
    # No access_levels field: which access levels a request may see is
    # computed server-side from the authenticated caller's clearance (see
    # app/core/security/access.py), never accepted from the client — see
    # docs/security.md "Retrieval-time enforcement".


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    text: str
    source_filename: str
    page_number: int | None
    section: str | None


class SearchResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    results: list[SearchResultItem]
