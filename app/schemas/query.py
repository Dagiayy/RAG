from pydantic import BaseModel, Field

from app.schemas.answer import GeneratedAnswerResponse
from app.schemas.graph import EmployeeMatchResponse
from app.schemas.search import SearchResultItem


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    evidence_top_k: int = Field(default=8, ge=1, le=50)


class QueryPlanResponse(BaseModel):
    use_graph: bool
    graph_query_type: str | None
    use_document_rag: bool
    reason: str


class QueryResponse(BaseModel):
    query: str
    intent: str
    entities: dict[str, str]
    plan: QueryPlanResponse
    answer: GeneratedAnswerResponse
    graph_matches: list[EmployeeMatchResponse]
    document_evidence: list[SearchResultItem]
