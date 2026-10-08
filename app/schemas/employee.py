from pydantic import BaseModel


class SkillResponse(BaseModel):
    name: str
    years_experience: int | None
    proficiency: str | None


class CertificationResponse(BaseModel):
    name: str
    issued_date: str | None
    expiry_date: str | None


class RoleResponse(BaseModel):
    title: str
    start_date: str | None
    end_date: str | None


class ProjectResponse(BaseModel):
    name: str
    role_on_project: str | None
    industry: str | None
    client: str | None


class EmployeeProfileResponse(BaseModel):
    pg_id: str
    full_name: str
    employee_code: str
    years_experience: int
    company_name: str | None
    department_name: str | None
    skills: list[SkillResponse]
    certifications: list[CertificationResponse]
    roles: list[RoleResponse]
    projects: list[ProjectResponse]


class BulkCVRequest(BaseModel):
    query: str
    format: str = "docx"  # "docx" | "pdf"
