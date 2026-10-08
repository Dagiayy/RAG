# Enterprise RAG + Knowledge Graph Platform

A local-first, production-oriented platform that answers natural-language
questions over enterprise data (documents, employees, projects, policies)
using hybrid retrieval (vector + BM25 + knowledge graph + structured SQL),
reranking, and grounded, cited generation via a local LLM (Ollama). Every
answer is evidence-backed, access-controlled, and explicit about missing or
conflicting information — it never fabricates enterprise facts.

This project is under active, phased development. See
`docs/implementation-status.md` for exactly what currently works.

## Architecture

Full write-up: [`docs/architecture.md`](docs/architecture.md).
Also see [`docs/data-model.md`](docs/data-model.md),
[`docs/retrieval-design.md`](docs/retrieval-design.md),
[`docs/security.md`](docs/security.md),
[`docs/evaluation.md`](docs/evaluation.md),
[`docs/deployment.md`](docs/deployment.md), and design decisions in
[`docs/decisions/`](docs/decisions/).

```
User → Streamlit UI → FastAPI → Auth → Query Understanding → Query Planner
     → {Vector (Qdrant) | BM25 | Knowledge Graph (Neo4j) | SQL (PostgreSQL)}
     → Hybrid Fusion (RRF) → Reranking (CrossEncoder) → Duplicate/Conflict
       handling → Evidence Assembly → LLM (Ollama) → Grounding Validation
     → Citations → Response
```

## Technology stack

Python 3.12+, FastAPI, PostgreSQL (SQLAlchemy/Alembic), Qdrant, Neo4j
Community, rank-bm25, sentence-transformers, HF CrossEncoder reranker,
Ollama, Streamlit, Docker Compose, pytest, ruff/black/mypy, GitHub Actions.

## Prerequisites

- Python 3.12+
- Docker Desktop (for PostgreSQL / Qdrant / Neo4j)
- [Ollama](https://ollama.com) installed and a model pulled, e.g.:
  ```
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,ingestion,retrieval,generation,documents,ui]"
copy .env.example .env
```

(bash equivalent: `python -m venv .venv && source .venv/bin/activate`)

## Configuration

Copy `.env.example` to `.env` and adjust. Nothing is hard-coded — models,
DB URLs, chunk sizes, retrieval K, reranker thresholds, and security
settings are all environment-driven (see `app/config/settings.py`).

## Running locally

```powershell
# start dependency services
docker compose up -d postgres qdrant neo4j

# start Ollama (separate terminal, or as a service) and confirm a model is pulled
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M

# run the API
.\scripts\dev.ps1 dev
# or: uvicorn app.main:app --reload --port 8010
```

(Port 8010, not the default 8000 — 8000 is already used by another project's
container on this dev machine. Change it freely if that's not true for you.)

Check it's alive:

```powershell
curl http://localhost:8010/health
curl http://localhost:8010/ready
```

Equivalent `make` targets exist for non-Windows shells (`make dev`, `make
docker-up`, etc.) — see the `Makefile`; `scripts/dev.ps1` is the Windows
equivalent (`.\scripts\dev.ps1 <target>`).

## Testing

```powershell
.\scripts\dev.ps1 test
# or: pytest tests/unit tests/integration -v
```

## Docker

```powershell
docker compose up -d --build
```

Brings up PostgreSQL, Qdrant, Neo4j, and the FastAPI app. Ollama is expected
to run on the host (see `docs/decisions/0002-ollama-host-process.md`).

## Ingesting documents

Apply migrations once, then upload a file (PDF, DOCX, TXT, Markdown, CSV,
JSON, or XLSX):

```powershell
alembic upgrade head

curl -X POST http://localhost:8010/documents/ingest -F "file=@path\to\doc.pdf"
curl http://localhost:8010/documents
curl http://localhost:8010/documents/<document_id>
```

Re-uploading byte-identical content returns `422` (exact-duplicate
rejection). Content that's substantially similar but not identical is
detected too (not blocked — both documents are kept, see
`near_duplicate_count` in the response). Topically-related content that
states a *different* numeric value for the same kind of quantity (a speed
limit, a budget figure, a day count, ...) is also detected — see
`conflicting_claim_count` in the response — via a deterministic regex +
embedding-similarity heuristic (no LLM call, so no hallucination risk;
see `app/services/ingestion/conflict_detection.py`). Both checks are
detection-only: nothing is ever blocked or discarded, findings are just
recorded for a human to review. See `docs/retrieval-design.md` for
chunking-strategy trade-offs.

Ingestion automatically embeds each chunk (`sentence-transformers/all-MiniLM-L6-v2`
by default) and indexes it into Qdrant.

**Versioning**: to upload a new version of an existing document, pass
`supersedes_document_uid=<the old document's document_uid>` — the old
version is marked superseded (kept, not deleted) and search prefers the
new one by default.

**Synthetic demo documents**: `scripts/seed_synthetic_documents.py` seeds
9 fictional documents permanently into the corpus — two independent
documents with genuinely conflicting facts (useful for seeing
`conflicting_evidence` answers for real), two near-duplicate documents,
and one document at each of the 5 access levels:

```powershell
python scripts/seed_synthetic_documents.py
```

Idempotent (checksum-deduped, same as any other ingest call).

## Authentication

`/search` and `/query` require a bearer token. Seed demo users at each
clearance level:

```powershell
python scripts/seed_users.py
```

This prints each user's token — use it as `Authorization: Bearer <token>`.
Access control is enforced server-side from the authenticated user's
clearance, inside the retrieval query itself (docs/security.md); it is
never accepted from the client.

## Searching (hybrid: vector + BM25, reranked)

```powershell
curl -X POST http://localhost:8010/search -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d "{\"query\": \"Siemens PLC experience\", \"top_k\": 5}"
```

Defaults to hybrid (vector + BM25, fused with Reciprocal Rank Fusion), then
reranked with a CrossEncoder over a wider candidate pool for the final
ordering. Pass `"mode": "vector"` or `"mode": "bm25"` to compare a single
retriever — useful since they cover different failure modes: BM25 finds
exact identifiers (project codes, model numbers) that a small embedding
model can miss; vector search finds paraphrases/semantic matches that BM25
misses. Pass `"rerank": false` to see pre-rerank ordering, or
`"include_historical": true` to include superseded document versions. See
`docs/retrieval-design.md`.

## Knowledge graph

Seed a small synthetic enterprise dataset (employees, skills, projects,
clients — see `scripts/seed_demo_data.py` for what it contains) into
PostgreSQL and Neo4j:

```powershell
python scripts/seed_demo_data.py
```

Then query the graph (requires a bearer token; results are clearance-
filtered the same way `/search` is — an employee's `access_level`
defaults to `internal`, see `app/models/enterprise.py`):

```powershell
curl "http://localhost:8010/graph/employees/by-skill?skill=GIS%20Mapping" -H "Authorization: Bearer <token>"
curl "http://localhost:8010/graph/employees/by-skill-and-industry?skill=Survey%20Design&industry=Humanitarian%20Aid" -H "Authorization: Bearer <token>"
curl "http://localhost:8010/graph/employees/by-certification?certification=Certified%20M%26E%20Professional" -H "Authorization: Bearer <token>"
curl "http://localhost:8010/graph/employees/by-client?client=AgriRise%20Foundation" -H "Authorization: Bearer <token>"
curl "http://localhost:8010/graph/projects/Refugee%20Camp%20Needs%20Assessment/team" -H "Authorization: Bearer <token>"
```

## Asking a question (grounded answer with citations)

```powershell
curl -X POST http://localhost:8010/query -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d "{\"query\": \"Which employees have GIS Mapping experience?\"}"
```

A local LLM (Ollama via its OpenAI-compatible API, `OPENAI_MODEL`) parses the question into a
structured intent (schema-constrained + Pydantic-validated — see
`docs/retrieval-design.md`), a planner decides whether that routes to a
graph query, hybrid document retrieval, or neither, and the response's
`answer` field contains a grounded prose answer with `[N]`-style citations,
a `confidence` ("evidence_supported" / "insufficient_evidence" /
"conflicting_evidence"), and the citation list — each citation is
guaranteed to reference real retrieved evidence, never fabricated (see
`app/services/generation/answer_generation.py`). If no evidence is found,
the LLM isn't even called: the response is the mandatory
`"I could not find sufficient authorized evidence to answer this question."`
The response also still includes the underlying `graph_matches`/
`document_evidence` for transparency alongside the answer. Both the
document-RAG path and the graph-matches path are access-controlled from
the caller's clearance.

## Generating CVs

```powershell
# full profile
curl http://localhost:8010/employees/<pg_id> -H "Authorization: Bearer <token>"

# single CV (docx or pdf)
curl -X POST "http://localhost:8010/employees/<pg_id>/cv?format=docx" -H "Authorization: Bearer <token>" -o cv.docx

# bulk: one CV per matching employee, returned as a zip
curl -X POST http://localhost:8010/employees/bulk-cv -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d "{\"query\": \"employees with GIS Mapping experience\", \"format\": \"pdf\"}" -o cvs.zip
```

Every CV is rendered directly from graph data — skills, certifications,
roles, and projects are real `HAS_SKILL`/`HAS_CERTIFICATION`/`HAS_ROLE`/
`WORKED_ON` relationships, never LLM-generated prose, so it cannot invent
experience. Each CV includes a traceability footer with the employee's
graph ID and generation timestamp. `pg_id` values come from `/graph/*` or
`/employees/{pg_id}` responses (or `scripts/seed_demo_data.py`'s output).
A CV can only be generated for an employee at or below the caller's
clearance — the same `access_level` check as every other graph query;
an unauthorized employee returns `404`, same as a nonexistent one.

## Running evaluation

A benchmark (`evaluation/benchmark.jsonl`) of real questions against the
seeded demo dataset — 10 graph-routed questions (skill/industry/
certification/client/project-team, ground-truthed against
`scripts/seed_demo_data.py`'s known employees) and 6 document-RAG
questions (ground-truthed against `evaluation/fixtures/docs/*.txt`, which
the run ingests temporarily and cleans up afterward so it never pollutes
the demo corpus):

```powershell
python scripts/seed_demo_data.py   # if not already seeded
python scripts/evaluate.py          # retrieval + answer-correctness metrics
python scripts/evaluate.py --judge  # also score each answer with the LLM judge
```

Computes, per spec: retrieval metrics (Recall@K/Precision@K/MRR/Hit
Rate/NDCG@K via `app/services/evaluation/retrieval_metrics.py`) and
generation metrics (answer correctness, citation precision/recall, context
precision via `app/services/evaluation/generation_metrics.py`) against
real benchmark ground truth, independent of each other so a bad prompt
can't mask good retrieval or vice versa. `--judge` additionally scores
each answer 1-5 on correctness/relevance/groundedness/citation quality
(`app/services/evaluation/llm_judge.py`) using the same local model as
generation — the only one available in this environment, a known
self-preference-bias risk (see Limitations) — writing results to
`evaluation/judge_results.jsonl`. A full report is written to
`evaluation/reports/<timestamp>.json` and a summary table is printed.

To check whether the judge's scores can be trusted, rate a sample
yourself and compare:

```powershell
python scripts/calibrate_judge.py --sample 10
```

Reports agreement rate and quadratic-weighted Cohen's kappa between your
scores and the judge's (interactive; needs `evaluation/judge_results.jsonl`
from a `--judge` run first).

## Security

See [`docs/security.md`](docs/security.md) for the threat model,
retrieval-time authorization enforcement, and prompt-injection defenses.

## Production deployment

See [`docs/deployment.md`](docs/deployment.md) for the local → single
server → internal enterprise → scalable production evolution path.

## Limitations (current)

- Document ingestion (PDF/DOCX/TXT/MD/CSV/JSON/XLSX, chunking, exact- and
  near-duplicate detection, versioning/supersedes) plus embedding + Qdrant
  vector indexing, BM25 lexical indexing, RRF-fused hybrid search, and
  CrossEncoder reranking all work end-to-end for document RAG — and
  `/search`/`/query`'s document-RAG path is genuinely access-controlled
  (bearer-token auth, server-computed clearance-based filtering enforced
  inside the retrieval query, verified by a security test that ingests
  restricted content and confirms a low-clearance user gets zero results
  referencing it — see `docs/implementation-status.md`). A small
  enterprise data model (employees/skills/projects/clients), a Neo4j
  knowledge graph with parameterized Cypher query templates, an LLM-backed
  query understanding + planning layer, and grounded answer generation
  with citations (`/query` — evidence-supported / insufficient-evidence /
  conflicting-evidence, citations that only ever reference real retrieved
  data), and CV generation (`/employees/{id}/cv`, `/employees/bulk-cv` —
  rendered directly from graph data, never LLM-generated, so it cannot
  invent experience) all work end-to-end. `/query`'s `comparison` intent
  still just gets a generic grounded answer over graph matches, not
  tailored comparison output. `/graph/*`, `/query`'s graph-matches path,
  and `/employees/*` (including CV generation) are now clearance-
  differentiated the same way document retrieval is — an `Employee`'s
  `access_level` (reusing `Document`'s same vocabulary) gates whether a
  caller can see that record anywhere in the graph-backed surface.
  `POST /documents/ingest` requires authentication, though it still
  doesn't check whether the caller's clearance justifies the
  `access_level` they're requesting for their own upload. There's still
  no entity extraction from ingested document text into the graph (a
  `Document`'s `access_level` and an `Employee`'s are set independently —
  nothing yet connects "this document mentions this employee" to the
  graph). There's also no
  unified document-delete pipeline yet: deleting from PostgreSQL doesn't
  cascade into Qdrant, and the graph isn't auto-synced on every write
  (`sync_all_to_graph()` must be called explicitly).
- Two local LLM models available in this environment
  (`qwen2.5:7b-instruct-q4_K_M` used for generation/query-understanding,
  `qwen2.5-coder:14b` also present but code-specialized), so LLM-as-judge
  evaluation still defaults to the same model as generation for now (see
  `docs/evaluation.md`) — a real self-preference-bias risk, not yet
  mitigated. The evaluation benchmark (`evaluation/benchmark.jsonl`, Phase
  11) covers graph-routed questions and a small set of synthetic document
  fixtures; it does not yet include cases for the conflicting-evidence/
  near-duplicate/restricted-document corpus (`scripts/
  seed_synthetic_documents.py`) — those scenarios have real integration
  test coverage (`tests/integration/test_synthetic_document_scenarios.py`,
  `tests/integration/test_answer_generation.py`) but aren't yet
  `evaluation/benchmark.jsonl` cases, since a substring-match
  `answer_correctness` check doesn't naturally express "did it correctly
  report both conflicting values instead of picking one."
  `answer_correctness` is a case-insensitive substring check against
  `expected_answer_contains`, not semantic similarity — a correct but
  differently-worded answer would score as incorrect; `--judge` is the
  intended complement for that gap, itself bounded by the single-model
  caveat above.

## Roadmap

Full phase-by-phase roadmap tracked in
[`docs/implementation-status.md`](docs/implementation-status.md).
