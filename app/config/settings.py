from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    app_debug: bool = True
    log_level: str = "INFO"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "enterprise_rag"
    postgres_user: str = "enterprise_rag"
    postgres_password: str = "change_me"
    database_url: str = (
        "postgresql+asyncpg://enterprise_rag:change_me@localhost:5432/enterprise_rag"
    )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "enterprise_chunks"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change_me"

    # LLM (OpenAI-compatible API — works against Ollama's /v1 endpoint, or
    # any other OpenAI-compatible provider, without code changes; only
    # these three settings change. See ADR 0005.)
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    openai_model: str = "qwen2.5:7b-instruct-q4_K_M"
    openai_judge_model: str = "qwen2.5:7b-instruct-q4_K_M"
    # Native Ollama API (not OpenAI-compatible) — only used for the /ready
    # liveness probe, which needs an endpoint that doesn't require a
    # chat-completion round trip to answer "is the server up".
    ollama_native_base_url: str = "http://localhost:11434"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Bumped manually when embedding_model (or preprocessing) changes, so
    # stored vectors can be identified as stale and re-indexed. HF model
    # names don't carry a reliable version string of their own.
    embedding_model_version: str = "v1"
    embedding_batch_size: int = 32
    embedding_cache_size: int = 2048

    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    # Retrieval
    retrieval_top_k_vector: int = 50
    retrieval_top_k_bm25: int = 50
    retrieval_rerank_candidates: int = 30
    retrieval_final_evidence_count: int = 8
    retrieval_min_evidence_score: float = 0.3
    retrieval_rrf_k: int = 60

    # Security
    max_upload_mb: int = 50

    # Observability
    request_id_header: str = "X-Request-ID"


@lru_cache
def get_settings() -> Settings:
    return Settings()
