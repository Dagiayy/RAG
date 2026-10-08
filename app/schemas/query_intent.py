from typing import Literal

from pydantic import BaseModel, Field

# Keys the extraction prompt is instructed to use; kept as a constant so the
# prompt and any validation logic can reference the same list.
ENTITY_KEYS = ("skill", "industry", "certification", "client", "project", "role")


class QueryFilter(BaseModel):
    field: Literal["years_experience"]
    operator: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float


class QueryIntent(BaseModel):
    """Structured form of a natural-language enterprise question. See
    docs/retrieval-design.md "Query understanding". Produced by an LLM
    (app/services/generation/query_understanding.py) but always validated
    against this schema before use — the raw LLM output is never executed
    or trusted directly, only used to parameterize typed retriever calls.
    """

    intent: Literal[
        "employee_search",
        "document_qa",
        "project_lookup",
        "cv_generation",
        "comparison",
        "general_qa",
    ]
    entities: dict[str, str] = Field(default_factory=dict)
    filters: list[QueryFilter] = Field(default_factory=list)
    requested_output: str = "answer"
