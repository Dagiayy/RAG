# Deployment Evolution

## Stage 1 — Local development (this repo's default)

`docker-compose.yml` brings up PostgreSQL, Qdrant, Neo4j; the FastAPI app and
Streamlit UI run via `make dev` (or Windows equivalents) against localhost
ports; Ollama runs as a host process. Single-user, no TLS, dev secrets in
`.env`, SQLite optionally usable for unit tests only.

## Stage 2 — Single server

Everything (including Ollama) containerized on one host via
`docker-compose.prod.yml`; a reverse proxy (Caddy/nginx) terminates TLS;
PostgreSQL/Qdrant/Neo4j get named volumes with a scheduled `pg_dump` /
Qdrant snapshot / Neo4j dump backup job (`scripts/backup.py` + cron).
Real OIDC-backed auth replaces the dev login endpoint behind the same
`AuthProvider` interface. Health/readiness endpoints wired to the proxy for
zero-downtime restarts.

## Stage 3 — Internal enterprise deployment

- **Database**: managed PostgreSQL (read replica for reporting/evaluation
  queries so they don't compete with request-path queries).
- **Vector DB**: Qdrant cluster mode (sharded collections) or managed Qdrant
  Cloud; collection-per-tenant if multi-department isolation is required
  beyond payload filtering.
- **Graph**: Neo4j causal cluster (read replicas) if graph query volume
  grows; Neo4j Community has no built-in clustering, so this is the trigger
  to move to Enterprise/Aura.
- **Model serving**: move from single-host Ollama to a dedicated inference
  service (e.g. vLLM/TGI behind an internal endpoint) for concurrency;
  Ollama's single-request-at-a-time behavior becomes the bottleneck.
- **Caching**: Redis for embedding cache and hot query results.
- **Queues**: ingestion moves from synchronous API calls to a task queue
  (e.g. Celery/RQ or Dramatiq) so large batch ingests don't block the API;
  status polled via `GET /documents/{id}`.
- **Secrets**: environment variables replaced by a secrets manager
  (Vault/cloud KMS); rotated, not baked into images.
- **Monitoring**: Prometheus + Grafana dashboards (latency per stage, error
  rate, retrieval quality drift); alerting on health-check failures.
- **Horizontal scaling**: FastAPI app becomes stateless (already is, if
  session state stays in PostgreSQL/Redis) behind a load balancer; multiple
  replicas.

## Stage 4 — Scalable production

- Multi-region read replicas for PostgreSQL/graph if latency to users
  demands it.
- Vector DB and graph DB scaled independently based on measured bottleneck
  (see `docs/evaluation.md` system metrics — don't scale blindly).
- Blue/green or canary deploys for model/prompt changes, gated by the
  evaluation benchmark (a prompt/model change must not regress the fixed
  benchmark before rollout).
- Disaster recovery: documented RPO/RTO, tested restore from backups
  (not just backups existing — restore drills).
- Full audit-log retention policy and access review process for compliance.

## What does NOT change across stages

The retrieval/generation pipeline code, the `AuthProvider`/`FusionStrategy`
interfaces, and the evaluation harness stay the same — only the
infrastructure behind fixed interfaces changes. This is the reason those
interfaces exist rather than concrete calls scattered through the codebase.
