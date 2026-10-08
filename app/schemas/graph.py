from pydantic import BaseModel


class EmployeeMatchResponse(BaseModel):
    pg_id: str
    full_name: str
    employee_code: str
    years_experience: int
    detail: str | None = None


class GraphQueryResponse(BaseModel):
    query_type: str
    results: list[EmployeeMatchResponse]
