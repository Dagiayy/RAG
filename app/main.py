from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.employees import router as employees_router
from app.api.graph import router as graph_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.api.search import router as search_router
from app.config import get_settings
from app.core.middleware import RequestContextMiddleware
from app.core.observability import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("startup")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("app_startup", env=settings.app_env)
        yield

    app = FastAPI(
        title="Enterprise RAG + Knowledge Graph Platform",
        version="0.1.0",
        description=(
            "Local-first enterprise knowledge intelligence platform combining "
            "vector search, BM25, a Neo4j knowledge graph, and structured "
            "PostgreSQL retrieval behind grounded, cited answer generation."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(graph_router)
    app.include_router(query_router)
    app.include_router(employees_router)

    return app


app = create_app()
