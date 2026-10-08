from fastapi import APIRouter, Depends, Query

from app.core.security.access import authorized_access_levels
from app.core.security.auth import get_current_user
from app.models.user import User
from app.schemas.graph import EmployeeMatchResponse, GraphQueryResponse
from app.services.graph import queries as graph_queries

router = APIRouter(prefix="/graph", tags=["graph"])


def _to_response(query_type: str, matches: list) -> GraphQueryResponse:
    return GraphQueryResponse(
        query_type=query_type,
        results=[
            EmployeeMatchResponse(
                pg_id=m.pg_id,
                full_name=m.full_name,
                employee_code=m.employee_code,
                years_experience=m.years_experience,
                detail=m.detail,
            )
            for m in matches
        ],
    )


@router.get("/employees/by-skill", response_model=GraphQueryResponse)
async def employees_by_skill(
    skill: str,
    min_years: int | None = Query(default=None, ge=0),
    current_user: User = Depends(get_current_user),
) -> GraphQueryResponse:
    levels = authorized_access_levels(current_user.clearance_level)
    matches = await graph_queries.employees_by_skill(skill, min_years, authorized_levels=levels)
    return _to_response("employees_by_skill", matches)


@router.get("/employees/by-industry", response_model=GraphQueryResponse)
async def employees_by_industry(
    industry: str, current_user: User = Depends(get_current_user)
) -> GraphQueryResponse:
    levels = authorized_access_levels(current_user.clearance_level)
    matches = await graph_queries.employees_by_industry(industry, authorized_levels=levels)
    return _to_response("employees_by_industry", matches)


@router.get("/employees/by-skill-and-industry", response_model=GraphQueryResponse)
async def employees_by_skill_and_industry(
    skill: str, industry: str, current_user: User = Depends(get_current_user)
) -> GraphQueryResponse:
    levels = authorized_access_levels(current_user.clearance_level)
    matches = await graph_queries.employees_by_skill_and_industry(
        skill, industry, authorized_levels=levels
    )
    return _to_response("employees_by_skill_and_industry", matches)


@router.get("/employees/by-certification", response_model=GraphQueryResponse)
async def employees_by_certification(
    certification: str, current_user: User = Depends(get_current_user)
) -> GraphQueryResponse:
    levels = authorized_access_levels(current_user.clearance_level)
    matches = await graph_queries.employees_by_certification(
        certification, authorized_levels=levels
    )
    return _to_response("employees_by_certification", matches)


@router.get("/employees/by-client", response_model=GraphQueryResponse)
async def employees_by_client(
    client: str, current_user: User = Depends(get_current_user)
) -> GraphQueryResponse:
    levels = authorized_access_levels(current_user.clearance_level)
    matches = await graph_queries.employees_by_client(client, authorized_levels=levels)
    return _to_response("employees_by_client", matches)


@router.get("/projects/{project_name}/team", response_model=GraphQueryResponse)
async def project_team(
    project_name: str, current_user: User = Depends(get_current_user)
) -> GraphQueryResponse:
    levels = authorized_access_levels(current_user.clearance_level)
    matches = await graph_queries.project_team(project_name, authorized_levels=levels)
    return _to_response("project_team", matches)
