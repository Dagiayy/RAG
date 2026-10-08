from typing import Literal

from pydantic import BaseModel


class CitationResponse(BaseModel):
    marker: int
    source_type: Literal["document", "employee"]
    document_id: str | None = None
    chunk_id: str | None = None
    source_filename: str | None = None
    page_number: int | None = None
    section: str | None = None
    employee_pg_id: str | None = None
    employee_name: str | None = None


class GeneratedAnswerResponse(BaseModel):
    answer: str
    confidence: Literal["evidence_supported", "insufficient_evidence", "conflicting_evidence"]
    citations: list[CitationResponse]
