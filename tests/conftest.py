import httpx
import pytest_asyncio

from app.config import get_settings
from app.repositories import db as db_module
from app.services.generation import llm_client as llm_client_module


def ollama_is_reachable() -> bool:
    """Used to skip LLM-dependent tests gracefully (rather than fail
    with a confusing connection error) when Ollama isn't running — e.g. in
    CI, which doesn't run a multi-GB local model. Real coverage of these
    tests still happens whenever Ollama is actually available (as it was
    for every run during Phase 7/8/9 development, see
    docs/implementation-status.md). Checks the native Ollama API (not the
    OpenAI-compatible one the app actually calls) since it's a plain GET
    with no request body, the simplest possible reachability probe.
    """
    try:
        response = httpx.get(f"{get_settings().ollama_native_base_url}/api/tags", timeout=1.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engine_after_test():
    """Forces a fresh engine/connection pool per test.

    Windows + asyncpg + ProactorEventLoop will try to clean up pooled
    connections against an event loop that's already closed if the engine
    (a process-wide singleton in app.repositories.db) outlives the loop it
    was created under. Disposing and clearing the cached engine after every
    test sidesteps that instead of chasing loop-scope pytest-asyncio config.
    """
    yield
    engine = db_module.get_engine()
    await engine.dispose()
    db_module._engine = None
    db_module._session_factory = None


@pytest_asyncio.fixture(autouse=True)
async def _reset_llm_client_after_test():
    """Same Windows per-test-event-loop issue as the engine fixture above,
    but for the cached AsyncOpenAI client (its underlying httpx transport
    binds to the event loop it was first used on): clear the cache so each
    test gets a fresh client bound to its own loop.
    """
    yield
    llm_client_module.get_llm_client.cache_clear()
