"""LLM client singleton, via the OpenAI-compatible chat completions API
(see ADR 0005). Points at Ollama's `/v1` endpoint by default
(`Settings.openai_base_url`), but swapping to a real OpenAI-compatible
provider (OpenAI itself, vLLM, TGI, etc. — see docs/deployment.md stage 3
"model serving") only ever needs `openai_base_url`/`openai_api_key`/
`openai_model` to change, never calling code. `openai.AsyncOpenAI` is
already async-native (unlike qdrant-client/neo4j/sentence-transformers
elsewhere in this codebase), so no `asyncio.to_thread` wrapping is needed.
"""

from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
