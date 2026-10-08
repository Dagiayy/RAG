import io

import docx
import fitz

from app.services.documents.cv_generator import render_cv_docx, render_cv_pdf
from app.services.graph.queries import (
    CertificationEntry,
    EmployeeProfile,
    ProjectEntry,
    RoleEntry,
    SkillEntry,
)


def _full_profile() -> EmployeeProfile:
    return EmployeeProfile(
        pg_id="test-pg-id-123",
        full_name="Test Employee",
        employee_code="TST-001",
        years_experience=7,
        company_name="Test Company",
        department_name="Test Department",
        skills=[SkillEntry(name="Python", years_experience=5, proficiency="expert")],
        certifications=[
            CertificationEntry(
                name="Test Certification", issued_date="2020-01-01", expiry_date="2026-01-01"
            )
        ],
        roles=[RoleEntry(title="Senior Engineer", start_date="2019-01-01", end_date=None)],
        projects=[
            ProjectEntry(
                name="Test Project",
                role_on_project="Lead",
                industry="Technology",
                client="Acme Corp",
            )
        ],
    )


def _minimal_profile() -> EmployeeProfile:
    return EmployeeProfile(
        pg_id="minimal-id",
        full_name="Minimal Employee",
        employee_code="MIN-001",
        years_experience=1,
        company_name=None,
        department_name=None,
        skills=[],
        certifications=[],
        roles=[],
        projects=[],
    )


def _docx_text(content: bytes) -> str:
    document = docx.Document(io.BytesIO(content))
    return "\n".join(p.text for p in document.paragraphs)


def _pdf_text(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def test_docx_contains_all_real_profile_data():
    text = _docx_text(render_cv_docx(_full_profile()))
    assert "Test Employee" in text
    assert "Python" in text
    assert "Test Certification" in text
    assert "Senior Engineer" in text
    assert "Test Project" in text
    assert "Acme Corp" in text
    assert "test-pg-id-123" in text  # traceability footer


def test_pdf_contains_all_real_profile_data():
    text = _pdf_text(render_cv_pdf(_full_profile()))
    assert "Test Employee" in text
    assert "Python" in text
    assert "Test Certification" in text
    assert "Senior Engineer" in text
    assert "Test Project" in text
    assert "test-pg-id-123" in text


def test_docx_handles_minimal_profile_without_error():
    text = _docx_text(render_cv_docx(_minimal_profile()))
    assert "Minimal Employee" in text
    assert "Skills" not in text  # no section header for empty lists
    assert "Certifications" not in text


def test_pdf_handles_minimal_profile_without_error():
    text = _pdf_text(render_cv_pdf(_minimal_profile()))
    assert "Minimal Employee" in text


def test_docx_and_pdf_never_invent_data_not_in_profile():
    # nothing in the minimal profile should produce fabricated skill/cert text
    docx_text = _docx_text(render_cv_docx(_minimal_profile()))
    pdf_text = _pdf_text(render_cv_pdf(_minimal_profile()))
    for forbidden in ("Python", "Test Certification", "Test Project"):
        assert forbidden not in docx_text
        assert forbidden not in pdf_text
