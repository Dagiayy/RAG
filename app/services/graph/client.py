"""Neo4j driver singleton. The neo4j Python driver is synchronous, so
callers on the async request path wrap calls in `asyncio.to_thread`
(consistent with how qdrant-client and sentence-transformers are handled
elsewhere in this codebase — see ADR pattern in app/services/retrieval).
"""

from functools import lru_cache

from neo4j import Driver, GraphDatabase

from app.config import get_settings


@lru_cache(maxsize=1)
def get_neo4j_driver() -> Driver:
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def close_neo4j_driver() -> None:
    get_neo4j_driver().close()
    get_neo4j_driver.cache_clear()
