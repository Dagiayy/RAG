import asyncio

import httpx
import structlog
from fastapi import APIRouter
from sqlalchemy import text

from app.config import get_settings
from app.repositories.db import get_engine
from app.services.graph.client import get_neo4j_driver
from app.services.retrieval.vector import get_qdrant_client

router = APIRouter(tags=["health"])
logger = structlog.get_logger("health")


@router.get("/health")
async def health() -> dict:
    """Liveness probe: the process is up. Does not check dependencies."""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict:
    """Readiness probe: can the app actually serve requests right now.

    Checks PostgreSQL, Qdrant, Neo4j, and the LLM backend.
    """
    checks: dict[str, str] = {}

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"
        logger.warning("readiness_check_failed", dependency="postgres", error=str(exc))

    try:
        await asyncio.to_thread(get_qdrant_client().get_collections)
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"
        logger.warning("readiness_check_failed", dependency="qdrant", error=str(exc))

    try:
        await asyncio.to_thread(get_neo4j_driver().verify_connectivity)
        checks["neo4j"] = "ok"
    except Exception as exc:
        checks["neo4j"] = f"error: {exc}"
        logger.warning("readiness_check_failed", dependency="neo4j", error=str(exc))

    try:
        settings = get_settings()
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            response = await http_client.get(f"{settings.ollama_native_base_url}/api/tags")
            response.raise_for_status()
        checks["llm"] = "ok"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"
        logger.warning("readiness_check_failed", dependency="llm", error=str(exc))

    overall_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if overall_ok else "degraded", "checks": checks}
