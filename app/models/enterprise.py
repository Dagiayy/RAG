"""Enterprise data model (see docs/data-model.md). PostgreSQL is the system
of record for attributes; Neo4j (app/services/graph) mirrors these rows as
graph nodes/relationships for traversal, keyed by `pg_id`.

Deliberately scoped to what Phase 6's graph queries actually use — Location,
Product, Equipment, Technology, Event from the conceptual model in
docs/data-model.md aren't implemented yet since nothing consumes them yet
(no query shape needs them). Add them when a real query shape does.
"""

import uuid
from datetime import date

from sqlalchemy import Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.document import AccessLevel


class Company(Base, TimestampMixin):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    is_own_org: Mapped[bool] = mapped_column(default=False)


class Department(Base, TimestampMixin):
    __tablename__ = "departments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    parent_department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("departments.id"), nullable=True
    )


class Industry(Base, TimestampMixin):
    __tablename__ = "industries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True)


class JobRole(Base, TimestampMixin):
    __tablename__ = "job_roles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(256), unique=True)


class Skill(Base, TimestampMixin):
    __tablename__ = "skills"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Certification(Base, TimestampMixin):
    __tablename__ = "certifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True)
    issuing_body: Mapped[str | None] = mapped_column(String(256), nullable=True)


class Employee(Base, TimestampMixin):
    __tablename__ = "employees"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_code: Mapped[str] = mapped_column(String(32), unique=True)
    full_name: Mapped[str] = mapped_column(String(256))
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    hire_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    years_experience: Mapped[int] = mapped_column(Integer, default=0)
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("departments.id"), nullable=True
    )
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True
    )
    # Reuses Document's AccessLevel vocabulary (app/models/document.py)
    # rather than inventing a second classification scheme — one clearance
    # ladder, enforced the same way (app/core/security/access.py) across
    # both documents and graph entities. See docs/security.md.
    access_level: Mapped[AccessLevel] = mapped_column(String(16), default=AccessLevel.INTERNAL)

    department: Mapped[Department | None] = relationship()
    company: Mapped[Company | None] = relationship()
    skills: Mapped[list["EmployeeSkill"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    certifications: Mapped[list["EmployeeCertification"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    roles: Mapped[list["EmployeeRole"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )
    projects: Mapped[list["EmployeeProject"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )


class EmployeeSkill(Base):
    __tablename__ = "employee_skills"
    __table_args__ = (UniqueConstraint("employee_id", "skill_id", name="uq_employee_skill"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE")
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("skills.id"))
    years_experience: Mapped[int] = mapped_column(Integer, default=0)
    proficiency: Mapped[str | None] = mapped_column(String(32), nullable=True)

    employee: Mapped[Employee] = relationship(back_populates="skills")
    skill: Mapped[Skill] = relationship()


class EmployeeCertification(Base):
    __tablename__ = "employee_certifications"
    __table_args__ = (
        UniqueConstraint("employee_id", "certification_id", name="uq_employee_certification"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE")
    )
    certification_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("certifications.id")
    )
    issued_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    employee: Mapped[Employee] = relationship(back_populates="certifications")
    certification: Mapped[Certification] = relationship()


class EmployeeRole(Base):
    __tablename__ = "employee_roles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE")
    )
    job_role_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("job_roles.id"))
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    employee: Mapped[Employee] = relationship(back_populates="roles")
    job_role: Mapped[JobRole] = relationship()


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256))
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True
    )
    client_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id"), nullable=True
    )
    industry_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("industries.id"), nullable=True
    )
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    company: Mapped[Company | None] = relationship(foreign_keys=[company_id])
    client: Mapped[Company | None] = relationship(foreign_keys=[client_id])
    industry: Mapped[Industry | None] = relationship()


class EmployeeProject(Base):
    __tablename__ = "employee_projects"
    __table_args__ = (UniqueConstraint("employee_id", "project_id", name="uq_employee_project"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("employees.id", ondelete="CASCADE")
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE")
    )
    role_on_project: Mapped[str | None] = mapped_column(String(256), nullable=True)

    employee: Mapped[Employee] = relationship(back_populates="projects")
    project: Mapped[Project] = relationship()
