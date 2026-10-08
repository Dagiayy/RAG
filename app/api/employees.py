import io
import zipfile

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security.access import authorized_access_levels
from app.core.security.audit import log_action
from app.core.security.auth import get_current_user
from app.models.user import User
from app.pipelines.cv_pipeline import select_cv_candidates
from app.repositories.db import get_db_session
from app.schemas.employee import (
    BulkCVRequest,
    CertificationResponse,
    EmployeeProfileResponse,
    ProjectResponse,
    RoleResponse,
    SkillResponse,
)
from app.services.documents.cv_generator import render_cv_docx, render_cv_pdf
from app.services.graph.queries import EmployeeProfile, get_employee_profile

router = APIRouter(prefix="/employees", tags=["employees"])

_CONTENT_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
}


def _profile_to_response(profile: EmployeeProfile) -> EmployeeProfileResponse:
    return EmployeeProfileResponse(
        pg_id=profile.pg_id,
        full_name=profile.full_name,
        employee_code=profile.employee_code,
        years_experience=profile.years_experience,
        company_name=profile.company_name,
        department_name=profile.department_name,
        skills=[
            SkillResponse(
                name=s.name, years_experience=s.years_experience, proficiency=s.proficiency
            )
            for s in profile.skills
        ],
        certifications=[
            CertificationResponse(name=c.name, issued_date=c.issued_date, expiry_date=c.expiry_date)
            for c in profile.certifications
        ],
        roles=[
            RoleResponse(title=r.title, start_date=r.start_date, end_date=r.end_date)
            for r in profile.roles
        ],
        projects=[
            ProjectResponse(
                name=p.name, role_on_project=p.role_on_project, industry=p.industry, client=p.client
            )
            for p in profile.projects
        ],
    )


def _render_cv(profile: EmployeeProfile, fmt: str) -> bytes:
    if fmt == "docx":
        return render_cv_docx(profile)
    if fmt == "pdf":
        return render_cv_pdf(profile)
    raise HTTPException(status_code=400, detail="format must be 'docx' or 'pdf'.")


@router.get("/{pg_id}", response_model=EmployeeProfileResponse)
async def get_employee(
    pg_id: str, current_user: User = Depends(get_current_user)
) -> EmployeeProfileResponse:
    """Full employee profile from the knowledge graph, clearance-filtered
    the same way `/graph/*` and `/query`'s graph path are (see
    docs/security.md) — an employee above the caller's authorized access
    level returns 404, the same response as a nonexistent pg_id, so
    existence isn't leaked to an unauthorized caller.
    """
    levels = authorized_access_levels(current_user.clearance_level)
    profile = await get_employee_profile(pg_id, authorized_levels=levels)
    if profile is None:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return _profile_to_response(profile)


@router.post("/{pg_id}/cv")
async def generate_employee_cv(
    pg_id: str,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
    format: str = Query(default="docx", pattern="^(docx|pdf)$"),
) -> Response:
    """Generates a CV for one employee, rendered directly from graph data
    (spec section 26) — never LLM-generated prose, so it cannot invent
    experience. See app/services/documents/cv_generator.py. Clearance-
    filtered the same way `get_employee` is: an unauthorized employee
    returns 404, not a CV.
    """
    levels = authorized_access_levels(current_user.clearance_level)
    profile = await get_employee_profile(pg_id, authorized_levels=levels)
    if profile is None:
        raise HTTPException(status_code=404, detail="Employee not found.")

    content = _render_cv(profile, format)
    await log_action(
        session, current_user, action="generate_cv", resource_type="employee", resource_id=pg_id
    )

    filename = f"{profile.full_name.replace(' ', '_')}_CV.{format}"
    return Response(
        content=content,
        media_type=_CONTENT_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/bulk-cv")
async def generate_bulk_cvs(
    request: BulkCVRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Natural-language candidate selection (spec section 26 example:
    "Generate CVs for all electrical engineers with more than 5 years of
    experience who have worked on cement projects") -> one CV per matching
    employee, returned as a ZIP archive. 404 if nothing matched (an empty
    ZIP would be a confusing silent success).
    """
    if request.format not in _CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="format must be 'docx' or 'pdf'.")

    levels = authorized_access_levels(current_user.clearance_level)
    profiles = await select_cv_candidates(request.query, authorized_levels=levels)
    if not profiles:
        raise HTTPException(status_code=404, detail="No employees matched the query.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for profile in profiles:
            content = _render_cv(profile, request.format)
            filename = f"{profile.full_name.replace(' ', '_')}_CV.{request.format}"
            zf.writestr(filename, content)

    await log_action(
        session,
        current_user,
        action="generate_bulk_cv",
        resource_type="employee",
        detail=f"query={request.query[:200]!r}, candidate_count={len(profiles)}",
    )

    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="cvs.zip"'},
    )
