from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.models.audit import AuditLog, IngestionAuditLog  # noqa: F401
from app.models.base import Base
from app.models.document import Document, DocumentChunk  # noqa: F401
from app.models.enterprise import (  # noqa: F401
    Certification,
    Company,
    Department,
    Employee,
    EmployeeCertification,
    EmployeeProject,
    EmployeeRole,
    EmployeeSkill,
    Industry,
    JobRole,
    Project,
    Skill,
)
from app.models.user import User  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_database_url() -> str:
    """Alembic runs synchronously; swap the asyncpg driver for psycopg."""
    settings = get_settings()
    return settings.database_url.replace("postgresql+asyncpg", "postgresql+psycopg")


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _sync_database_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
