# Implementation Status

Last updated: 2026-10-01 (Phase 11 complete)

## Completed phases

- **Phase 0 — Architecture**: `docs/architecture.md`, `docs/data-model.md`,
  `docs/retrieval-design.md`, `docs/security.md`, `docs/evaluation.md`,
  `docs/deployment.md`, `docs/decisions/0001`–`0004` written. Project
  directory scaffold created. Git repo initialized.

- **Phase 1 — Project Foundation**: `pyproject.toml` (deps split into
  optional extras: ingestion/retrieval/generation/documents/ui/dev),
  `.env.example`/`.env`, pydantic-settings config (`app/config`), structlog
  JSON logging (`app/core/observability`), request-ID middleware
  (`app/core/middleware.py`), FastAPI app with lifespan startup
  (`app/main.py`), `/health` + `/ready` endpoints (Postgres check wired into
  `/ready`), async SQLAlchemy engine (`app/repositories/db.py`),
  `docker-compose.yml` (postgres/qdrant/neo4j/app), `docker/Dockerfile`,
  GitHub Actions CI (`.github/workflows/ci.yml`, lint+format+typecheck+
  unit+integration against a Postgres service container), pre-commit config,
  Makefile + `scripts/dev.ps1` (Windows equivalent), `.gitignore`, initial
  README.

- **Phase 2 — Document Ingestion**: SQLAlchemy models (`app/models/document.py`:
  `Document`/`DocumentChunk` with checksum-unique constraint, `SourceType`/
  `DocumentStatus`/`AccessLevel` enums; `app/models/audit.py`:
  `IngestionAuditLog`), Alembic wired to async settings via a sync
  psycopg URL swap (`migrations/env.py`), initial migration applied.
  Ingestion services: `validation.py` (extension allowlist, magic-byte
  sniffing for PDF/DOCX/XLSX, size limit, path-traversal-safe filenames),
  `extraction.py` (PDF via PyMuPDF, DOCX via python-docx, TXT/MD, CSV, JSON,
  XLSX via openpyxl — all real, no stubs), `chunking.py` (fixed/sentence/
  paragraph/heading_aware strategies, semantic chunking explicitly deferred
  to Phase 3 pending the embedding service — not faked), `storage.py`
  (raw file persistence under `data/raw/{document_uid}/`), `pipeline.py`
  (orchestrates validate→dedupe-by-checksum→extract→chunk→persist→audit).
  API: `POST /documents/ingest`, `GET /documents`, `GET /documents/{id}`
  (`app/api/documents.py`).

- **Phase 3 — Embeddings + Qdrant**: `app/services/embeddings/service.py`
  (`EmbeddingService`: configurable model via `Settings.embedding_model`,
  batching, L2 normalization, in-memory LRU cache keyed by content hash,
  bounded retry with backoff, process-wide model cache + a
  `get_embedding_service()` singleton so the text cache is actually shared
  across requests). `app/services/retrieval/vector.py` (Qdrant collection
  management, payload built from the full chunk-metadata list in
  `docs/data-model.md`, `search()` with metadata filters + score threshold,
  `delete_document_vectors()`). Ingestion pipeline now embeds + indexes
  every chunk on ingest (`asyncio.to_thread` since both the model and
  qdrant-client are sync), stamping `embedding_model`/
  `embedding_model_version` on each `DocumentChunk` row. New
  `POST /search` endpoint (vector-only — explicitly documented as NOT
  access-controlled yet, pending Phase 8's AuthProvider). `/ready` now also
  checks Qdrant. `scripts/benchmark_embeddings.py` measures real throughput/
  latency (see Tests passed below — no fabricated numbers).

- **Phase 4 — BM25 + Hybrid Retrieval**: `app/services/retrieval/bm25.py`
  (`BM25Index` wrapping rank-bm25's `BM25Okapi`; simple lowercase
  alphanumeric tokenizer so exact identifiers aren't stemmed away; the
  index is a process-wide singleton rebuilt lazily from PostgreSQL on next
  use after being invalidated by ingestion, rather than rebuilt on every
  query or eagerly on every ingest — see module docstring for the
  trade-off). `app/services/retrieval/fusion.py` (`FusionStrategy`
  Protocol + `ReciprocalRankFusion`, matching ADR 0003). New
  `app/services/retrieval/hybrid.py` (`hybrid_search()` runs vector +
  BM25 concurrently and fuses with RRF; `run_search()` dispatches
  `mode="hybrid"|"vector"|"bm25"` to the same result shape so callers can
  compare retrievers). `/search` now defaults to hybrid mode with
  `mode` as an explicit override; ingestion invalidates the BM25 index
  after every successful ingest.

- **Phase 5 — Reranking**: `app/services/reranking/service.py`
  (`RerankerService` wrapping a HuggingFace `CrossEncoder`, configurable
  via `Settings.reranker_model`, bounded retry with backoff, process-wide
  singleton `get_reranker_service()` — module docstring explains why a
  CrossEncoder is a separate, expensive-but-accurate second stage rather
  than the primary retriever). `run_search()` (`app/services/retrieval/hybrid.py`)
  now fetches a wider candidate pool (`Settings.retrieval_rerank_candidates`,
  default 30) from whichever mode was selected, reranks with the
  CrossEncoder, and truncates to `top_k` — controlled by a new `rerank`
  parameter (default `True`), orthogonal to `mode`. `/search` exposes
  `rerank: bool` and returns `rerank_score` alongside `vector_score`/
  `bm25_score`. `scripts/benchmark_reranking.py` compares vector-similarity
  ordering against CrossEncoder ordering on a deliberately ambiguous
  5-candidate case and reports real latency (see Tests passed below).

- **Phase 6 — Knowledge Graph (Neo4j)**: enterprise data model added to
  PostgreSQL (`app/models/enterprise.py`: `Company`, `Department`,
  `Industry`, `Skill`, `Certification`, `JobRole`, `Employee` +
  `EmployeeSkill`/`EmployeeCertification`/`EmployeeRole`/`EmployeeProject`
  join tables, `Project`; migrated via Alembic) — this didn't exist before
  Phase 6 (only `documents`/`document_chunks` did), and the graph needs
  real entities to traverse. `app/services/graph/client.py` (Neo4j driver
  singleton), `schema.py` (uniqueness constraints on `pg_id` per label),
  `sync.py` (`sync_all_to_graph()`: idempotent MERGE-based sync from
  Postgres, sync-order respects FK dependencies, relationship properties
  carried over e.g. `HAS_SKILL.years_experience`), `queries.py` (6
  parameterized Cypher templates matching spec section 13's example
  question shapes — `employees_by_skill`, `employees_by_industry`,
  `employees_by_skill_and_industry` [multi-hop], `employees_by_certification`,
  `employees_by_client`, `project_team` — per ADR 0004, no free-form
  LLM-generated Cypher). New `GET /graph/employees/by-*` and
  `GET /graph/projects/{name}/team` endpoints. `/ready` now also checks
  Neo4j. `scripts/seed_demo_data.py` seeds a small (8-employee) synthetic
  dataset — domain flavor (field data collection / monitoring-evaluation /
  IoT-for-development work) loosely inspired by real-world firms in that
  space (looked at greenproffs.com's public site for realistic
  department/role/skill categories), but every company, employee, and
  project name is fictional; no real organization's data or any real
  person's identity was used. The full synthetic document dataset
  (conflicting versions, duplicates, restricted docs) landed later —
  see "Post-Phase-11 work" below.

- **Phase 7 — Graph + RAG (query planning)**: `app/schemas/query_intent.py`
  (`QueryIntent`/`QueryFilter` Pydantic schema — the LLM's raw output is
  never executed, only validated then used to parameterize typed retriever
  calls). `app/services/generation/ollama_client.py` (Ollama's first real
  use in this project — `AsyncClient` singleton). `app/services/generation/
  query_understanding.py` (`understand_query()`: uses Ollama's structured-
  output `format=<json schema>` decode-time constraint + Pydantic
  validation + one retry with the validation error fed back + a hard
  fallback to `general_qa` if the LLM is unreachable/unreliable — verified
  this actually degrades gracefully, not just in theory). `app/pipelines/
  query_planner.py` (`plan_query()`: entity-presence-based routing —
  skill+industry → multi-hop graph template, skill/industry/certification/
  client alone → the matching single-hop template, project → project-team
  template, no graph-queryable entities or a document/general question →
  hybrid retrieval; never runs both when one suffices). `app/pipelines/
  query_pipeline.py` (`run_query()`: understand → plan → execute →
  post-filter graph matches by any `years_experience` filter — combines
  everything from Phases 4-6 behind one call). New `POST /query` endpoint.
  **Scope note**: this returns structured evidence (graph matches and/or
  reranked document chunks), NOT a generated prose answer — grounded
  generation with citations is Phase 9.

- **Phase 8 — Enterprise Features**. Prioritized within this phase (see
  "Known issues" below for what's explicitly deferred and why):
  - **Access control** (spec section 21) — `app/models/user.py` (`User`,
    `UserRole`, `ClearanceLevel`), `app/core/security/auth.py`
    (`AuthProvider` protocol + `StaticUserAuthProvider`: dev-mode bearer
    token lookup against the `users` table — a later production stage
    swaps in real OIDC behind the same protocol without touching
    callers), `app/core/security/access.py` (clearance ↔ `AccessLevel`
    ordering/comparison). `/search` and `/query` now **require**
    authentication and compute the caller's authorized access levels
    server-side from their clearance — the client can no longer supply
    `access_levels` at all (removed from the request schema entirely, not
    just ignored). Enforced inside the retrieval query itself (Qdrant/BM25
    payload filter), not by post-hoc filtering. `scripts/seed_users.py`
    seeds 4 demo users spanning all 4 clearance levels.
  - **Document versioning** (spec section 20) — `ingest_document()` gained
    `supersedes_document_uid`: looks up the previous version, sets its
    `status=SUPERSEDED` in **both** PostgreSQL and Qdrant (a real bug
    caught here: Qdrant's payload is an ingest-time snapshot, not
    live-synced with Postgres — a status update needs an explicit
    `update_document_payload()` call, added to `vector.py`), and links
    `supersedes_id`/`version`. `run_search()` gained `include_historical`
    (default `False`): retrieval prefers the current version by default,
    superseded versions remain queryable explicitly, never deleted.
  - **Near-duplicate detection** (spec section 18) — `app/services/
    ingestion/duplicates.py` (`find_near_duplicate_chunks()`: searches
    Qdrant for each new chunk's nearest neighbors above a similarity
    threshold, run before the new chunks are upserted so there's no
    self-match risk). Detection only — never blocks ingestion or discards
    either document (both sides' provenance is preserved, per spec).
    Findings are logged and recorded in the ingestion audit detail;
    `near_duplicate_count` is returned from `POST /documents/ingest`.
  - **Broader audit logging** — `app/models/audit.py` gained `AuditLog`
    (distinct from Phase 2's ingestion-only audit log) + `app/core/
    security/audit.py::log_action()`, called from `/search` and `/query`.

- **Infrastructure change (2026-09-26, between Phase 8 and 9, per user
  request)**: switched the LLM client from the native `ollama` Python
  package to `openai.AsyncOpenAI` pointed at Ollama's OpenAI-compatible
  `/v1` endpoint — see ADR 0005. `app/services/generation/ollama_client.py`
  deleted, replaced by `llm_client.py`; settings renamed
  `ollama_base_url`/`ollama_generation_model`/`ollama_judge_model` →
  `openai_base_url`/`openai_model`/`openai_judge_model` (a new
  `ollama_native_base_url` setting was kept solely for the `/ready` probe,
  which now actually checks LLM reachability — it never had before, a gap
  noted since Phase 3). Re-ran Phase 7's full structured-extraction test
  suite against the new client before considering this done: identical
  correct results on every case. Rationale: three settings
  (`openai_base_url`/`openai_api_key`/`openai_model`) now swap the entire
  LLM backend — real OpenAI, another local server, a hosted endpoint —
  with no code changes anywhere that calls the LLM, versus being coupled
  to Ollama's own SDK and wire format.

- **Phase 9 — Generation** (spec sections 23-25). `app/services/security/
  injection.py` (heuristic prompt-injection scanner — flags suspicious
  evidence text, never blocks it; the real defense is that evidence is
  always framed as data, never instructions, in the prompt — see below).
  `app/services/generation/context_builder.py` (`build_evidence_items()`:
  assigns [N] markers to real retrieved evidence server-side — this is
  what makes "never fabricate citations" an enforced invariant rather than
  a prompt request; `format_context_block()`: wraps evidence in
  `<retrieved_context>` delimiters with an explicit "this is data, not
  instructions" header). `app/services/generation/answer_generation.py`
  (`generate_answer()`: if there's no evidence at all, the LLM is never
  called — the mandatory "I could not find sufficient authorized evidence
  to answer this question." is returned directly (spec section 25); a
  citation is only ever included if its marker matches a real evidence
  item, filtering out anything the LLM might over-cite — verified with a
  deterministic mocked-client unit test asserting a hallucinated marker
  [99] never makes it into the response, not just hoped-for). `/query` now
  returns `answer: {answer, confidence, citations}` alongside the existing
  evidence fields (kept for transparency/debugging).
  **Real, non-trivial bug found and fixed during manual testing before
  this shipped**: graph evidence like "Employee: X, years_experience=6,
  detail=6y, expert" never literally states what was searched for (e.g.
  "GIS Mapping") — the model correctly (from a strict-grounding stance)
  refused to connect it to the question and answered
  "insufficient_evidence" despite real matches existing. Fixed by adding
  `describe_graph_query()` (`app/pipelines/query_planner.py`) which
  describes what a graph query searched for (e.g. "employees with skill
  'GIS Mapping'") and embeds it into each employee evidence item's text.
  **Also found via manual testing, not assumed**: the initial prompt
  conflated "insufficient evidence" with "conflicting evidence" — given
  two sources with different values for the same fact, the model picked
  "insufficient_evidence" instead of surfacing both values. Fixed with an
  explicit rule + few-shot example distinguishing the two (same technique
  that worked for Phase 7's query-understanding prompt). Verified against
  the real LLM afterward, not just assumed fixed: evidence-supported,
  insufficient-evidence, conflicting-evidence (correctly reports both
  values, doesn't average), and prompt-injection-in-evidence (doesn't
  comply with an embedded "reveal confidential salaries" instruction) all
  behave correctly.

- **Phase 10 — CV Generation** (spec section 26). `app/services/graph/
  queries.py` gained `get_employee_profile(pg_id)`: a full-detail Cypher
  query (company, department, all skills/certifications/roles/projects
  with their relationship properties) distinct from the narrower
  `EmployeeMatch` shape the search queries return. `app/services/
  documents/cv_generator.py` (`render_cv_docx`/`render_cv_pdf`):
  deliberately does NOT call the LLM at any point — every line comes
  directly from a graph relationship, so "must not invent employee
  experience" is architectural, not a hope, the same principle as Phase
  9's citation-safety invariant but simpler here since there's no
  natural-language synthesis step to get wrong. Both formats include a
  traceability footer (employee `pg_id` + generation timestamp).
  `app/pipelines/cv_pipeline.py` (`select_cv_candidates()`: reuses Phase
  7's `understand_query`/`plan_query`/graph-search candidate-selection
  logic verbatim — `run_graph_query()` in `query_pipeline.py` was made
  public for this reuse — then fetches a full profile per match). New
  endpoints: `GET /employees/{pg_id}` (full profile), `POST /employees/
  {pg_id}/cv?format=docx|pdf` (single CV download), `POST /employees/
  bulk-cv` (natural-language query → one CV per match, returned as a ZIP;
  404 if nothing matched rather than a silent empty archive). All three
  require authentication as a baseline (same documented gap as `/graph/*`:
  no clearance-based filtering on employee data yet, since the entity
  model has no classification field). **Deliberately not built**:
  `POST /employees/search` from spec section 30 — `GET /graph/employees/
  by-*` (Phase 6) already serves this need; a separate endpoint duplicating
  the same graph queries under a different path was judged not worth the
  duplication.

- **Phase 11 — Evaluation** (spec section 27). `app/services/evaluation/
  retrieval_metrics.py` (`compute_retrieval_metrics()`: Recall@K,
  Precision@K, MRR, Hit Rate, NDCG@K from a ranked id list + a
  ground-truth relevant-id set — generic over document-chunk ids and graph
  `pg_id`s so the same harness scores both retrieval paths;
  `average_metrics()` aggregates across a benchmark run). `app/services/
  evaluation/generation_metrics.py` (`answer_correctness()`: case-
  insensitive substring check against `expected_answer_contains` — a
  simple, honest baseline, not semantic similarity, see Known issues;
  `context_precision()`/`citation_precision()`/`citation_recall()`: how
  much of what was retrieved/cited was actually ground-truth relevant).
  `app/services/evaluation/llm_judge.py` (`judge_answer()`: a second
  Ollama call scoring correctness/relevance/groundedness/citation_quality
  1-5 with a rubric prompt, same structured-output + Pydantic-validation
  pattern as Phase 7/9; degrades to `None` rather than raising on
  failure). `app/services/evaluation/calibration.py`
  (`weighted_cohens_kappa()`/`calibrate()`: quadratic-weighted Cohen's
  kappa between judge and human scores — a real statistical
  implementation, not a stub, verified against hand-computable cases: 1.0
  for perfect agreement, 0.0 for a degenerate-marginal case matching its
  own chance baseline, negative when two non-degenerate raters disagree
  worse than chance would predict).

  `evaluation/benchmark.jsonl`: 16 real cases — 10 graph-routed questions
  ground-truthed against `scripts/seed_demo_data.py`'s known employees
  (by stable `employee_code`, resolved to `pg_id` at eval time, not
  hardcoded UUIDs), and 6 document-RAG questions ground-truthed against 4
  new fixture documents (`evaluation/fixtures/docs/*.txt`: a remote-work
  policy, a drought-monitoring field protocol, an IoT sensor calibration
  guide, an ESG verification checklist — each with distinct, checkable
  numeric/factual content, e.g. "calibrated every 90 days"). `scripts/
  evaluate.py` orchestrates a run: ingests the fixture documents, runs
  every benchmark case through the real `run_query()` pipeline (same path
  `/query` uses), computes retrieval + generation metrics against ground
  truth, optionally runs the LLM judge (`--judge`), writes a full report
  to `evaluation/reports/<timestamp>.json`, and — critically — **deletes
  the fixture documents again afterward** (Postgres row + Qdrant vectors +
  BM25 index invalidation), the same track-then-clean-up pattern this
  project's integration tests use throughout. This was necessary, not
  just tidy: `tests/integration/test_query_pipeline.py` has an existing
  test asserting "What is our policy on remote work?" returns
  `insufficient_evidence` because no documents exist yet — permanently
  ingesting a real remote-work-policy fixture would have silently broken
  that test's premise. Verified by running the full suite (191/191,
  unchanged assumption) immediately after a real `evaluate.py` run, not
  assumed safe. `scripts/calibrate_judge.py`: interactive CLI reading
  `evaluation/judge_results.jsonl`, hides the judge's own score while the
  human rates each case (avoids anchoring bias), reports agreement rate +
  weighted kappa.

  **Real run against the live stack** (2026-10-01, `python
  scripts/evaluate.py --judge`, all 16 cases, Postgres + Qdrant + Neo4j +
  Ollama): 16/16 scored, **100% answer accuracy** (substring match),
  **retrieval @ k=10: recall=1.00, precision=0.20, mrr=1.00, hit_rate=1.00,
  ndcg=1.00** (precision@10 < 1 is expected and correct — most ground-truth
  answer sets have far fewer than 10 relevant items, e.g. a 1-2-employee
  skill match, so most of the top-10 slots are necessarily non-relevant;
  recall/mrr/ndcg at 1.0 is the metric that actually matters here and
  confirms every relevant item was found and correctly ranked), **judge
  coverage 16/16**, judge scores mostly 4-5 across all four dimensions with
  plausible rationales (spot-checked, not just trusted — e.g. one case
  correctly dinged for an answer adding an unrequested detail, another for
  a missing explicit citation). Confirmed both Postgres and Qdrant back to
  0 rows/points after cleanup via direct query, not assumed. This is a
  small, high-precision-ground-truth benchmark by design (Phase 9's full
  synthetic dataset with conflicting/duplicate/restricted documents isn't
  built yet — see Known issues below), so a 100%/1.00 result reflects
  "the implemented paths work correctly end-to-end on known-answerable
  questions," not "the system is perfect" — it doesn't yet stress-test
  ambiguous, adversarial, or partially-answerable cases.

## Post-Phase-11 work

- **Synthetic document dataset** (the gap flagged since Phase 6).
  `scripts/seed_synthetic_documents.py` permanently seeds 9 fictional
  documents into the demo corpus (idempotent — checksum dedup, same as
  every other ingest call): 2 independent (not supersedes-linked)
  documents giving different numeric answers to the same real-world
  question (vehicle speed limit: 60 km/h per a 2023 policy vs. 40 km/h
  per a 2024 addendum), 2 near-duplicate documents (reworded equipment
  checkout procedure — real run confirmed `near_duplicate_count=1` fired
  on ingest, not assumed), and one document at each of the 5
  `AccessLevel` values (public holiday calendar through executive
  compensation) so clearance-based filtering has real content at every
  level, not just the single PUBLIC/RESTRICTED pair
  `tests/security/test_retrieval_leakage.py` already covered.

  **Required a pre-existing test fix first**:
  `tests/integration/test_query_pipeline.py`'s
  `test_pipeline_document_question_routes_to_hybrid_search` asserted "What
  is our policy on remote work?" returns `insufficient_evidence` because
  no documents existed in the corpus at test time — true before this
  seed script, false after. Fixed by switching the question to a
  uuid-suffixed fictional policy name (same isolation pattern used
  throughout this suite for graph entities), so the test's "nothing
  matches" premise holds regardless of what else gets permanently seeded,
  not just at the moment the test was written. Verified unaffected
  (194/194) after the real synthetic corpus was seeded.

  **Real, non-trivial bug found via manual testing before this shipped**
  (same rigor as Phase 9's original grounding/conflict-confusion bugs —
  not just "it worked once, ship it"): with the real conflicting
  documents actually retrieved end-to-end (both in the top 2 of 8
  evidence items by rerank score), the model self-resolved the conflict
  by inferring the newer-dated "addendum" document superseded the older
  "policy" document, reporting `evidence_supported` instead of
  `conflicting_evidence` — even though the two documents are NOT linked
  via the system's real supersedes mechanism (Phase 8's
  `supersedes_document_uid`), so nothing in the evidence actually
  establishes which one is authoritative. This reproduced with two
  independently-worded document pairs (an explicit "2023 policy" /"2024
  addendum" pair and an abstract "original policy"/"updated addendum"
  pair with a nonsense topic name), and the first prompt fix (one added
  rule + one few-shot example) only fixed the first pair, not the
  second — confirming it was pattern-matching the example's surface
  wording rather than generalizing the rule. Fixed for real by (1)
  rewriting the rule in `SYSTEM_PROMPT`
  (`app/services/generation/answer_generation.py`) to explicitly name the
  trigger words ("updated", "addendum", "revised", "newer", "supersedes",
  "original", "previous", "former") and state that recency language is
  part of the disagreement, not a resolution of it, and (2) adding a
  SECOND few-shot example with different wording from the first
  (original/updated framing instead of dated-policy framing) so the model
  has two differently-surfaced instances of the same underlying rule to
  generalize from. Re-verified against the real LLM: both document pairs
  now correctly report `conflicting_evidence` with both values cited,
  confirmed stable across 3 repeated runs (temperature=0).

  New permanent regression coverage added (not just a one-off manual
  check): `tests/integration/test_answer_generation.py::
  test_conflicting_evidence_is_not_self_resolved_via_recency` (hand-built
  evidence, fast, isolates the prompt behavior) and
  `tests/integration/test_synthetic_document_scenarios.py` (2 tests: the
  same scenario through REAL retrieval — embedding, Qdrant, BM25, RRF,
  reranking, generation, not hand-built evidence objects — plus a
  full-ladder access-control test covering all 5 `AccessLevel` values,
  extending the existing PUBLIC/RESTRICTED-only security test). Full
  suite: 194/194 passing, lint/format clean.

- **Conflict detection** (spec section 19, deferred since Phase 8 — see
  the Phase 8 entry above). `app/services/ingestion/conflict_detection.py`
  implements the numeric-heuristic approach the Phase 8 deferral note
  explicitly named as the safer alternative to LLM-based claim
  extraction: `extract_numeric_claims()` pulls NUMBER+UNIT pairs out of
  chunk text via regex (e.g. "60 km/h", "$420,000", "90 days" — a bare
  number with no unit word/symbol right after it is never extracted, since
  it's too ambiguous to compare safely), and `find_conflicting_claims()`
  reuses the same embedding-similarity search `duplicates.py` already
  uses for near-duplicate detection to find topically-related existing
  chunks, then flags a pair only when both sides have a claim with the
  SAME unit but a DIFFERENT number. No LLM call anywhere in this path —
  fully deterministic, auditable (every finding names the exact two
  numbers/unit/chunks), and detection-only (never blocks ingestion, never
  auto-resolves, never discards either document — same posture as
  near-duplicate detection). Wired into `ingest_document()`
  (`app/services/ingestion/pipeline.py`) right alongside near-duplicate
  detection, reusing the same already-computed chunk vectors; surfaced as
  `conflicting_claim_count` on `IngestionResult`,
  `DocumentIngestResponse` (API), and the ingestion audit log detail.

  **Real design bug found and fixed before this shipped** (caught by its
  own integration test failing, not assumed correct): the first version
  excluded any match above near-duplicate detection's 0.92 similarity
  threshold from conflict consideration, reasoning "that's near-duplicate
  territory, not a conflict." This was backwards — two sentences
  differing ONLY in their numeric claim ("the limit is 60 units per hour"
  vs. "the limit is 40 units per hour") score ABOVE 0.92 on sentence
  embeddings (a single digit token barely moves the embedding), so the
  exclusion was hiding exactly the clearest, most confident kind of
  conflict instead of catching it. Fixed by removing the upper-bound
  exclusion entirely — near-duplicate and conflict detection are
  independent, complementary signals (a pair can legitimately be BOTH
  near-identically worded AND numerically conflicting), not mutually
  exclusive categories. Verified via 3 integration tests against the real
  stack: different values for the same topic ARE flagged (incl. the
  near-identical-wording case that exposed the bug), the same value
  restated is NOT flagged, and unrelated documents sharing a coincidental
  number but different units are NOT flagged. Plus 6 unit tests for the
  regex extraction itself (multi-claim text, currency, percent as both
  symbol and word, unit normalization, bare-number rejection). Full
  suite: 203/203 passing, lint/format clean.

  Not yet re-verified against the permanent synthetic corpus
  (`scripts/seed_synthetic_documents.py`'s vehicle-speed-limit pair) with
  conflict detection active, since it predates this feature and
  re-seeding would require deleting the existing corpus first — a
  bulk-delete action the harness's safety classifier correctly declined
  to run unprompted (`DELETE FROM documents` / a filterless Qdrant points
  delete). Not necessary for confidence (the isolated integration tests
  above exercise the identical code path against real Postgres+Qdrant),
  but if the permanent corpus is ever rebuilt, this would be confirmed
  then too.

- **Graph-level and document-ingest access control** (the design decision
  flagged as deferred since Phase 8 — "deciding what classification
  should mean — department-based? explicit per-project classification?").
  Resolved by reusing `Document`'s existing `AccessLevel` vocabulary
  (public/internal/department/confidential/restricted) rather than
  inventing a second scheme: one clearance ladder, enforced the same way,
  across both documents and graph entities. `app/models/enterprise.py`'s
  `Employee` gained an `access_level` column (migration
  `59e79e7ed8d1`, `server_default="internal"` since the table already had
  seeded rows — a plain `nullable=False` add would have failed against
  them, unlike `Document.access_level`'s original migration which started
  from an empty table). `Project` was deliberately NOT given its own
  classification field — every graph query ultimately returns
  `EmployeeMatch`/`EmployeeProfile` objects, so gating at the Employee
  node uniformly covers every query shape (by-skill, by-industry,
  by-certification, by-client, project-team) without a second field to
  keep in sync; a project's visibility is implicitly gated through
  whichever employees' records a caller is authorized to see.

  `app/services/graph/sync.py` now writes `access_level` onto every
  Employee node. Every employee-returning function in `app/services/
  graph/queries.py` (`employees_by_skill`, `_by_industry`,
  `_by_skill_and_industry`, `_by_certification`, `_by_client`,
  `project_team`, `get_employee_profile`) gained an `authorized_levels:
  list[str] | None = None` parameter — `None` means unrestricted (used by
  internal callers: scripts, tests that don't go through an authenticated
  API path), filtered via `e.access_level IN $authorized_levels` inside
  the Cypher query itself, same retrieval-time-enforcement principle
  docs/security.md already established for documents.
  `get_employee_profile()` returns `None` (not a 403) for an employee that
  exists but isn't authorized — same response shape as "doesn't exist", so
  existence isn't leaked. Threaded through both pipelines that reach the
  graph (`run_graph_query()`/`run_query()` in `app/pipelines/
  query_pipeline.py`, `select_cv_candidates()` in `app/pipelines/
  cv_pipeline.py`) and all three API surfaces that call them:
  `/graph/*` (previously fully unauthenticated — now requires a bearer
  token, same as `/search`/`/query`), `/employees/*` (previously
  authenticated-but-unfiltered — now clearance-filtered), and `/query`'s
  graph-matches path (previously unfiltered — now computes
  `authorized_access_levels()` for the graph path exactly like it already
  did for the document-RAG path). `POST /documents/ingest` also gained
  the `get_current_user` dependency (previously the one fully
  unauthenticated write endpoint) plus an audit-log entry recording who
  uploaded what — it still does not check whether the caller's clearance
  justifies the `access_level` they're requesting for the upload itself,
  which is a separate, more nuanced policy question left open.

  Verified end-to-end against the real stack, not just unit-level:
  `tests/security/test_graph_access_control.py` (4 tests) — `/graph/*`
  rejects unauthenticated requests; a PUBLIC-clearance user querying for a
  RESTRICTED-access employee's skill gets an empty result while a
  RESTRICTED-clearance user gets the real match; `/employees/{pg_id}` and
  `/employees/{pg_id}/cv` both 404 for an unauthorized employee and
  succeed for an authorized one; `/query`'s `graph_matches` and citations
  never reference a restricted employee for a low-clearance caller (same
  citation-layer extension pattern as Phase 9's document-side security
  test). `tests/integration/test_documents_api.py` (2 tests) — ingest
  without credentials is rejected, ingest with credentials succeeds and is
  audited. Pre-existing demo data (`scripts/seed_demo_data.py`'s 8
  employees, all defaulting to `internal`) re-synced cleanly and manually
  spot-checked live: a PUBLIC-clearance token gets zero `/graph/*` results
  for a real skill (correctly excluded — PUBLIC can't see `internal`-level
  records), a DEPARTMENT-clearance token gets the real matches. Full
  suite: 209/209 passing, lint/format clean.

## Current phase

- **Further post-Phase-11 work** (not started). Remaining items: a custom
  (non-Streamlit, per explicit user direction) UI, entity extraction from
  ingested document text into the graph, and human calibration of the LLM
  judge. See "Next tasks" below for what's recommended first.

## Environment detected

- OS: Windows 11 (win32), shell PowerShell/Git Bash both available.
- Python 3.13.1, Docker 28.3.2 + Compose v2.38.2, git 2.43.0, Node 20.15.0.
- Ollama CLI present but daemon not running at inspection time (will need
  `ollama serve` / app running before generation features work).

## Tests passed

- `pytest tests/` (unit + integration + security + evaluation) →
  191/191 passed (all Ollama-dependent tests, incl. Phase 9/10/11's, ran
  for real in this dev environment rather than self-skipping — Ollama was
  reachable throughout; two separate full-suite runs during Phase 7/8
  development each hit one transient Ollama hiccup under load, confirmed
  not a regression by re-running in isolation — see Known issues): settings
  defaults,
  `/health` incl. request-ID header, chunking (all 4 strategies + edge
  cases), extraction (all 7 formats incl. real generated PDF/DOCX/XLSX
  fixtures), validation (extension/magic-byte/size/path-traversal), hashing,
  embedding service (batching, L2 normalization, caching incl. LRU
  eviction, retry-then-succeed, retry exhaustion — against a fake model
  double, no network/model download needed for these), reranker service
  (relevance-based ordering, top-k, retry-then-succeed, retry exhaustion —
  same fake-model-double pattern), RRF fusion (formula correctness,
  weighting, multi-retriever combination, sort order), BM25 index
  (tokenization, exact-term ranking, zero-score exclusion, top-k, metadata
  lookup), Qdrant vector store (upsert+search, access-level metadata
  filter, document deletion, search-on-missing-collection), live-Postgres
  ingestion pipeline (create+chunks, exact-duplicate rejection, empty-file
  rejection), live-Postgres+Qdrant ingestion-with-real-embeddings
  (downloads the actual all-MiniLM-L6-v2 model, verifies chunks are
  embedding-stamped and semantically searchable), live hybrid search (BM25
  finds an exact project code across a realistic multi-doc corpus, hybrid
  fusion surfaces the right chunk, access-level filter excludes restricted
  content from hybrid results), live reranking (rerank=True populates
  `rerank_score` and rerank=False doesn't, reranked top result is the
  genuinely correct document, scores are sorted descending), live Neo4j
  graph sync (node+relationship creation incl. multi-hop
  Employee→Project→Industry, idempotent re-sync produces no duplicate
  nodes), live graph query templates (all 6 query shapes return exactly
  the expected employees against an isolated test graph, `min_years`
  filtering on the `HAS_SKILL` relationship property, unknown-skill query
  returns empty rather than erroring), query planner routing (13 cases —
  every intent×entity combination routes to the right template or
  correctly falls back to document RAG), `QueryIntent`/`QueryFilter` schema
  validation (rejects unknown intent/field/operator values), live query
  understanding against the real Ollama model (skill extraction, multi-hop
  skill+industry+experience-filter extraction, project-name extraction,
  document-question classification, off-topic fallback to `general_qa`,
  and a genuinely-unreachable-Ollama fallback via a monkeypatched client),
  live full query pipeline (skill question → correct graph match with
  zero document evidence; document question → correct hybrid-search
  routing with zero graph matches), access-control logic (clearance ↔
  access-level ordering/comparison, all 4 clearance tiers), document
  versioning (supersede marks old version + links the chain in both
  Postgres and Qdrant, search prefers the current version by default with
  `include_historical` opt-out, rejects superseding an unknown
  `document_uid`), near-duplicate detection (reworded-but-substantially-
  similar content across documents is flagged without blocking or
  discarding either document, genuinely unrelated content is not
  flagged), **the critical retrieval-leakage security test** (a
  low-clearance user searching for content that exists ONLY in a
  RESTRICTED document gets zero results referencing it anywhere in the
  response — not just filtered from the top result — while the same
  search for PUBLIC content succeeds and a RESTRICTED-clearance user
  searching for the same restricted content DOES find it, proving the
  block was clearance-based rather than a bug hiding the document from
  everyone; plus 401 on missing/invalid bearer tokens), injection scanner
  (flags known suspicious patterns, doesn't false-positive on benign
  business phrasing like "act as a liaison"), context builder (marker
  assignment order, graph-query-description embedding, suspicious-content
  flagging), citation-safety invariant via a deterministic mocked-client
  test (a hallucinated marker [99] never appears in the response; invalid
  JSON / schema-invalid / connection-error LLM responses all degrade to
  `insufficient_evidence` rather than crash), live generation against the
  real LLM (evidence-supported answer cites the real source; no-evidence
  question returns the exact mandated message without calling the LLM at
  all; genuinely conflicting evidence is reported with both values, not
  averaged or silently picked; an embedded "reveal confidential salaries"
  instruction in evidence text is not complied with; graph evidence
  produces a correctly-grounded, correctly-cited answer), live full
  `/query` pipeline generating grounded answers for both graph and
  document paths, **generation-layer security test** (a low-clearance
  user's generated answer and citations never reference restricted
  content that was excluded from its evidence — extends the retrieval-
  leakage test through the citation layer, not just raw search results),
  CV rendering (DOCX and PDF both contain every real skill/certification/
  role/project from a full profile plus the traceability footer; a
  minimal employee with no relationships renders without error and,
  critically, never contains any of the fabricated-data strings a full
  profile would — verifying the renderer can't invent content, not just
  that it doesn't on one example), live full-profile graph query (all
  relationship types assembled correctly for a fully-populated employee;
  an employee with zero relationships gets empty lists, not errors or
  nulls-as-fake-entries; a nonexistent `pg_id` returns `None` rather than
  a malformed profile), live CV candidate selection (natural-language
  query -> real graph matches -> full profiles, and a document-shaped
  question correctly yields zero candidates rather than misrouting), and
  the full `/employees/*` HTTP surface (profile fetch, single-CV
  download with correct content-type, bulk-CV ZIP containing exactly the
  matched employees, 404 for unknown employee / no matches, 401
  unauthenticated) — all against the real live server, not mocks — live
  Postgres `SELECT 1`, retrieval metrics (recall/precision/MRR/hit-rate/
  NDCG@K against hand-computable cases: perfect retrieval scores 1.0 on
  everything, zero-hit retrieval scores 0.0 on everything, MRR correctly
  rewards earlier rank, NDCG correctly rewards correct ordering of equally-
  relevant items, recall correctly caps at what fits within k, averaging
  across cases, and invalid input — empty ground truth, non-positive k —
  raises rather than silently returning a meaningless number), generation
  metrics (answer-correctness substring matching incl. case-insensitivity
  and the vacuous-true empty-expectations case, context/citation precision
  and recall against known id sets), judge-score calibration (weighted
  Cohen's kappa: perfect agreement -> 1.0, a degenerate-marginal case
  matches its own chance baseline exactly as the math predicts — not just
  "looks reasonable", small score gaps score higher agreement than large
  ones when isolating gap size from marginal distribution, mismatched-
  length/empty input raises), live LLM-as-judge against the real Ollama
  model (a clearly-correct answer scores higher on correctness than a
  clearly-wrong one for the same question — the property the whole
  mechanism depends on; an unreachable/misconfigured judge model degrades
  to `None` rather than raising).
- `ruff check .` → all checks passed (migrations/versions excluded from
  lint — autogenerated).
- `black --check .` → all files formatted.
- `python scripts/benchmark_embeddings.py` (real run, 2026-09-25, CPU-only
  on this dev machine): 160 sentences embedded in 0.63s (**~255 texts/sec**,
  batch_size=32, dim=384); Qdrant upsert of 160 points in 0.21s; search
  latency **mean 28.2ms / max 32.4ms** over 3 queries. Semantic relevance
  spot-checked manually — top result for "Siemens PLC experience" was
  correctly the Siemens sentence (score 0.651); "financial reporting"
  correctly matched the financial sentence (0.454). Not a calibrated
  recall/precision benchmark (that needs Phase 11's labeled dataset) — just
  a throughput/latency/sanity check.
- `python scripts/benchmark_reranking.py` (real run, 2026-09-25, CPU-only,
  models warmed up before timing so load time isn't conflated with
  inference): on a deliberately ambiguous 5-candidate case (a chunk sharing
  the query's surface words but off-topic, vs. an on-topic chunk with
  different wording), both vector-only and CrossEncoder ranking agreed on
  the correct top-1 result, but the CrossEncoder was far more decisive
  (score 8.26 for the right chunk vs. -7.26 for the next one — a clean
  separation — versus cosine similarities bunched at 0.72/0.41/0.22/0.22).
  Mid-ranking order changed between the two methods. Vector similarity over
  5 candidates: 125.2ms; CrossEncoder rerank over the same 5: 68.0ms. Not a
  labeled precision benchmark — a qualitative "does reranking sharpen
  discrimination" + latency check, honestly reported either way.
- Manual smoke tests: `uvicorn` boots; `GET /health`/`GET /ready` (now
  incl. Qdrant + Neo4j, all `"ok"`) OK; full HTTP round-trip of
  `POST /documents/ingest` → `GET /documents` → `GET /documents/{id}` →
  re-upload same file → `422 Duplicate content`; `POST /search`
  end-to-end in all three modes (`hybrid`/`vector`/`bm25`) against a
  live-ingested document; `GET /graph/employees/by-skill`,
  `by-skill-and-industry`, and `/graph/projects/{name}/team` against the
  real seeded dataset, results matched hand-computed expectations exactly;
  `POST /query` end-to-end for both a graph-routed question ("Which
  employees have GIS Mapping experience?" → correct 2 employees, empty
  document evidence) and a document-routed question (correctly classified
  `document_qa`, empty evidence since nothing ingested about it at the
  time) — all verified via curl against the live server + Postgres +
  Qdrant + Neo4j + Ollama; structured JSON request logs with `request_id`
  confirmed in stdout. Phase 8: unauthenticated `POST /search` → `401`;
  `python scripts/seed_users.py` → 4 demo users printed with tokens,
  re-run confirmed idempotent (same tokens returned, no duplicates);
  ingested v1 then v2 of a document via `supersedes_document_uid`,
  authenticated `POST /search` with `mode=vector` returned ONLY v2's
  chunk, not v1's — real versioning behavior confirmed over the live
  authenticated HTTP API, not just at the service-function level. Phase 9:
  `GET /ready` now shows `"llm":"ok"` (previously never checked — a gap
  noted since Phase 3); authenticated `POST /query` against the real
  seeded dataset returned a fully-formed grounded answer with correct
  citations pointing at real employee `pg_id`s, verified end-to-end over
  the live HTTP API, not just at the service-function level. Phase 10:
  authenticated `GET /employees/{pg_id}` returned the real, complete
  profile for the seeded "Michael Owusu"; `POST /employees/{pg_id}/cv` in
  both formats returned real, correctly-sized DOCX (37KB) and PDF (2.4KB)
  files whose content was read back and verified (not just a byte-count
  check); `POST /employees/bulk-cv` for "employees with GIS Mapping
  experience" returned a real ZIP.
- `python scripts/seed_demo_data.py`: real run, seeded 8 employees/4
  projects/5 companies/9 skills/3 certifications/7 job roles into
  Postgres, synced 62 relationships into Neo4j; re-run confirmed
  idempotent (identical node counts, no duplicates).
- CI workflow (`.github/workflows/ci.yml`) updated with Qdrant + Neo4j
  service containers (Neo4j healthcheck via `cypher-shell`) +
  `ingestion`/`retrieval`/`generation` extras (previously only installed
  `[dev]`, which would have failed on `test_extraction.py`'s real
  PyMuPDF/python-docx/openpyxl fixtures since Phase 2 — an unfixed gap
  until now) + an HF-cache step so the embedding model isn't re-downloaded
  every run + (Phase 8) an `alembic upgrade head` step, since CI had never
  actually run migrations before — a real gap that would have failed
  immediately on first push (every integration test assumes tables exist)
  — plus a `pytest tests/security` step + (Phase 10) the `documents`
  extra, since the new CV tests import `python-docx`/`reportlab` and CI
  didn't install them — caught by checking `pip install` extras against
  what the new tests actually need, not discovered by a failed run since
  this never ran on GitHub. Not yet run on GitHub (no remote pushed yet)
  — will confirm green on first push.

## Known issues / environment notes

- Removed `Settings.jwt_secret` (Phase 8): dead config left over from an
  early guess in Phase 1 before auth was actually designed. The real
  implementation (Phase 8) uses opaque per-user `api_key` bearer tokens
  looked up in Postgres, not JWTs — `jwt_secret` had zero consumers.
  Removed from `settings.py`, `.env`, and `.env.example` rather than left
  as unused/misleading config.
- Ollama became intermittently slow/unresponsive ("Server disconnected
  without sending a response") during two separate long (3-5 minute) full-
  suite test runs in this environment (Phase 7 and Phase 8), each time on
  exactly one Ollama-calling test, recovering immediately afterward with
  no action taken. Plausible cause: this machine has two local Ollama
  models available (`qwen2.5:7b-instruct-q4_K_M` + `qwen2.5-coder:14b`)
  competing for RAM alongside sentence-transformers, a CrossEncoder,
  Postgres, Qdrant, and Neo4j all running simultaneously during a full
  suite run — not confirmed, just the most likely explanation. Both times,
  the graceful-fallback behavior in `understand_query()` (Phase 7) did
  exactly what it's designed to do: logged the failure and fell back to
  `general_qa` rather than crashing the test/request. Re-running the
  affected test in isolation passed cleanly both times. Treat a lone
  Ollama-related failure in a full suite run as this known flakiness
  first — check `curl localhost:11434/api/tags` and re-run in isolation —
  rather than assuming a regression, but don't dismiss a *repeated*
  failure on the same test without investigating further.
- Ollama confirmed installed with model `qwen2.5:7b-instruct-q4_K_M` pulled
  (per user, 2026-09-25). Set as default `OLLAMA_GENERATION_MODEL`. A
  second model, `qwen2.5-coder:14b`, was found available in this
  environment as of Phase 7 (2026-09-26) — code-specialized, not an ideal
  general judge model semantically, so `OLLAMA_JUDGE_MODEL` still defaults
  to the same generation model for now; worth reconsidering once Phase 11
  (evaluation/LLM-as-judge) actually needs a distinct judge model.
  Structured extraction in Phase 7 used `qwen2.5:7b-instruct-q4_K_M` with
  Ollama's `format=<json schema>` decode-time constraint and got every
  test case right on the first real run (see Tests passed above) — a
  genuinely good result for a 7B local model, not a foregone conclusion.
- Both Docker Desktop and the Ollama daemon had stopped by the start of
  Phase 7 (new day/session — 2026-09-26 vs. 2026-09-25), taking the
  `enterprise-rag-*` containers down with them. Both needed to be
  restarted (`Start-Process ... Docker Desktop.exe` + `docker compose up
  -d postgres qdrant neo4j` + `ollama serve` in background) before work
  could continue. Data survived intact in named volumes across the
  restart (verified: 8 employees in Postgres, 44 nodes in Neo4j,
  unchanged). Expect this after any machine restart/idle period — check
  `docker ps` and `curl localhost:11434/api/tags` early in a new session
  rather than assuming yesterday's running state.
- Same Windows-per-test-event-loop issue as the async Postgres engine
  (Phase 1/2 note below), now also hit the cached `ollama.AsyncClient`
  singleton (its httpx transport binds to the event loop of its first
  request) — surfaced as `RuntimeError: Event loop is closed` on the
  *second* Ollama-calling test in a run, not the first, which is a classic
  sign of this exact bug class. Fixed the same way: an autouse
  `tests/integration/conftest.py` fixture clears
  `get_ollama_client.cache_clear()` after every test. Pattern to remember
  for any future cached async client singleton: it must not outlive the
  event loop it was created under, and pytest-asyncio's default
  function-scoped loop guarantees it won't survive past one test unless
  explicitly disposed.
- CI can't practically run the Ollama-dependent tests (would need a
  multi-GB model pulled in the CI runner). Rather than skip them from the
  test suite entirely or let them fail confusingly on a connection error,
  `tests/integration/conftest.py::ollama_is_reachable()` probes
  `{OLLAMA_BASE_URL}/api/tags` and the two Ollama-dependent test files use
  `pytestmark = pytest.mark.skipif(not ollama_is_reachable(), ...)` — a
  real, honestly-labeled skip, not a fake pass. They ran for real
  (unskipped) throughout Phase 7 development in this environment.
- Found a real (if minor) test-design flakiness, not a pipeline bug:
  `test_pipeline_routes_skill_question_to_graph_and_returns_match` used a
  random gibberish skill name (e.g. `Skill-a1b2c3d4`, from the test's
  uniqueness helper) in a phrasing like "Which employees have
  Skill-a1b2c3d4 experience?" — the LLM would sometimes not recognize an
  arbitrary-looking token as a skill entity (unlike realistic names like
  "GIS Mapping"), occasionally classifying it as `general_qa` instead of
  `employee_search`. Confirmed by reproducing in isolation (passed) vs. in
  the full suite (failed once). Fixed by rephrasing the test query to
  explicitly label the value ("the skill named '...'"), removing the
  ambiguity rather than relying on the LLM inferring intent from a
  meaningless string's surface form; reran 5/5 clean after the fix. Worth
  remembering for any future test that generates random unique names and
  feeds them through an LLM extraction step: keep the *role* of the value
  unambiguous in the prompt, independent of whether the value itself looks
  realistic.
- This dev machine already runs containers/ports for several other local
  projects. Conflicts found and resolved: Postgres host port remapped
  5432→**5433** (`docker-compose.yml`, `.env.example`), app dev port
  remapped 8000→**8010** (Makefile, `scripts/dev.ps1`, README, Compose
  `app` service). Qdrant (6333/6334) and Neo4j (7474/7687) had no conflicts.
  `docker compose up` can get killed by the harness if host memory is
  critically low at that moment — if it happens again, do not blindly
  retry; check `docker stats`/`docker system df` first (this repo's images
  are small; the low-memory event above was environmental, not caused by
  this project).
- `mypy` is wired into CI as non-blocking (`|| true`) for now per spec
  section 44 ("where practical") — codebase is still tiny; will tighten
  once more modules exist.
- Windows + asyncpg + pytest-asyncio: the process-wide cached async engine
  (`app/repositories/db.py`) must not outlive the event loop it was created
  under, or connection cleanup throws `RuntimeError: Event loop is closed`
  on Windows' ProactorEventLoop. Fixed with an autouse fixture in
  `tests/integration/conftest.py` that disposes and clears the cached
  engine after every test. Not an issue for the running app (one loop for
  the whole process lifetime), only for per-test loop churn.
- `document.department` is a plain nullable string, not yet a FK to a
  `departments` table — that table doesn't exist until Phase 8's broader
  enterprise data model lands (see `app/models/document.py` docstring).
- Pinned `qdrant-client` to `<1.16` and the Compose `qdrant`/`neo4j` images
  to exact versions (`v1.15.0` / `5.26-community`) instead of `latest` —
  found a real client/server version-mismatch warning (client 1.19.1 vs
  server 1.15.0) during testing; pinning both sides avoids silent drift
  between what's installed locally and what Compose pulls next time.
- Found and fixed a real bug during Phase 3: `upsert_chunks()` didn't
  accept/forward a `collection_name` parameter, so it always wrote to the
  default collection regardless of what callers configured via
  `ensure_collection(...)`. Caught by integration tests using a separate
  test collection; fixed by threading `collection_name` through
  `upsert_chunks()` (`app/services/retrieval/vector.py`). The same bug
  existed in `scripts/benchmark_embeddings.py`'s call site — also fixed.
- There is no unified "delete a document" pipeline yet: deleting a
  `Document` row in PostgreSQL does NOT cascade into Qdrant (separate
  stores). Callers must explicitly call
  `delete_document_vectors(document_id)` too. A proper delete endpoint/
  pipeline that keeps Postgres+Qdrant+Neo4j in sync is deferred — likely
  Phase 8 territory alongside versioning/supersedes.
- Near-duplicate/semantic duplicate detection and version/supersedes
  linking are NOT implemented yet (Phase 2 only does exact-checksum
  duplicate rejection) — tracked for Phase 8.
- Ingestion pipeline embeds + indexes into Qdrant and invalidates the BM25
  index (Phase 3/4), but does not yet extract entities into Neo4j — that's
  Phase 6.
- `/search` (hybrid by default now) is explicitly NOT access-controlled yet
  (see its docstring and `docs/security.md`) — do not point it at real
  confidential content until Phase 8.
- Found and fixed a second real test-hygiene bug during Phase 4:
  `tests/integration/test_ingestion_pipeline.py`'s cleanup only deleted the
  Postgres `Document` row, never the corresponding Qdrant vectors (it
  predates Phase 3's embedding indexing). Every run of that test file left
  orphaned vectors behind — 9 had silently accumulated in the
  `enterprise_chunks` collection by the time this was caught, via a manual
  `/search` smoke test surfacing stale "Duplicate content for testing..."
  results as noise, not by the tests themselves. Fixed by switching that
  file to the same track-and-delete-on-teardown fixture pattern already
  used in `test_ingestion_with_embeddings.py`/`test_hybrid_search.py`;
  verified the full suite now leaves both stores at 0 rows/points after a
  run.
- BM25 correctly returns zero results when a term's document-frequency
  ratio is too high for a tiny corpus (classic BM25 IDF degeneracy — e.g.
  a single-document corpus always scores 0, since a term appearing in
  100% of documents carries no discriminating information under BM25's
  IDF formula). This is real, correct BM25 behavior, not a bug in
  `app/services/retrieval/bm25.py` — confirmed by the same query working
  correctly once a handful of unrelated documents are present (see
  `tests/integration/test_hybrid_search.py` and
  `tests/unit/test_bm25.py`, both of which needed >1-document corpora for
  exactly this reason). Worth knowing before assuming `/search?mode=bm25`
  is broken against a near-empty demo corpus.
- `run_search()` defaults to `rerank=True`, which changes the effective
  behavior of any existing caller using `mode="bm25"`/`mode="vector"`
  without an explicit `rerank` argument (it now also reranks). Caught while
  reviewing `test_bm25_finds_exact_code_vector_search_may_rank_low`, whose
  whole point was to demonstrate raw BM25 retrieval — fixed by passing
  `rerank=False` there explicitly so the test still tests what its name
  says. Other callers (the `/search` API) intentionally keep the new
  `rerank=True` default since that's the better out-of-the-box behavior;
  just something to be deliberate about, not implicit, when writing new
  tests or scripts against `run_search()`.
- First draft of `scripts/benchmark_reranking.py` measured model-loading
  time (several seconds, one-time) as if it were inference latency,
  because — unlike `benchmark_embeddings.py`, which incidentally warms the
  model via its `.dimension` property access before starting the timer —
  nothing triggered the CrossEncoder's lazy load before the timed block.
  Fixed by explicitly warming both models first; corrected numbers (125ms/
  68ms) are in Tests passed above. A reminder that "first call includes
  lazy model load" is an easy way to produce misleading benchmark numbers
  in this codebase generally.

- Phase 6 required PostgreSQL tables that didn't exist yet — Phases 0-5
  only ever needed `documents`/`document_chunks` (a document-RAG-only
  data model). The full enterprise data model (`app/models/enterprise.py`)
  had to be built now rather than in a dedicated later phase, since a
  knowledge graph needs actual entities (employees/skills/projects) to
  traverse; docs/data-model.md already documented this schema
  conceptually, so this was "build what was already designed," not a
  design change. Location/Product/Equipment/Technology/Event from that
  doc's conceptual model are still NOT implemented — deferred until a real
  query shape needs them (see `app/models/enterprise.py` docstring).
- `document.department` (a plain string, see earlier note) and
  `Employee.department_id` (a proper FK to the new `departments` table)
  are two independent, unreconciled representations of "department" —
  documents and employees don't share a department taxonomy yet. Fixing
  this — i.e., giving `Document` a real `department_id` FK — is Phase 8
  territory alongside the rest of the enterprise-features work.
- The Neo4j graph is a derived index, not synced automatically on every
  Postgres write yet: `sync_all_to_graph()` must be called explicitly
  (currently only from `scripts/seed_demo_data.py`). Document ingestion
  does NOT yet extract entities/relationships into the graph — spec
  section 6's "entity extraction → entity resolution → knowledge graph
  update" ingestion pipeline steps aren't implemented. Correction to a
  note written here during Phase 6: it predicted this would land in
  "Phase 7" — Phase 7 turned out to be query understanding/planning per
  the master spec's actual phase list, not document entity extraction.
  That work has no assigned phase number in the original 14-phase
  breakdown; it's a genuine gap between spec section 6 (ingestion) and
  section 42 (phases) worth flagging rather than silently leaving
  unscheduled. Entity extraction from free document text is also a
  meaningfully different, harder problem (NER + entity resolution against
  existing graph nodes) than syncing already-structured employee/project
  rows, so it deserves explicit scoping whenever it's picked up.

- Found and fixed a real API-misuse bug in `update_document_payload()`
  (new in Phase 8): qdrant-client's `set_payload()` takes the point
  selector as a `points` argument accepting a `Filter` directly, not a
  `points_selector=FilterSelector(filter=...)` wrapper (that's the
  parameter shape for `delete()`, not `set_payload()`) — caught
  immediately by the versioning integration test failing with a
  `TypeError`, not silently shipped.
- Found a second FK-ordering test-cleanup bug (same class as the Phase 6
  Neo4j one): `test_document_versioning.py`'s cleanup fixture deleted
  documents in creation order, but a superseding document's
  `supersedes_id` FK references the superseded (older, created-first) row
  — Postgres correctly refused to delete the referenced parent while the
  child still existed. Fixed by deleting in reverse creation order.
  General lesson reconfirmed: any test that creates entities with FK
  relationships between them must clean up child-before-parent, not just
  "in the order created."
- **Explicitly deferred within Phase 8** (not silently dropped — flagging
  what's NOT done so it isn't assumed to be):
  - **Conflict detection** (spec section 19) — deferred at the time this
    note was written; implemented later (see "Post-Phase-11 work" below)
    using the numeric-heuristic approach named here as the safer
    alternative to LLM-based claim extraction.
  - **Graph-level access control** — deferred at the time this note was
    written (no classification field existed on `Employee`); implemented
    later (see "Post-Phase-11 work" below) by reusing `Document`'s
    `AccessLevel` vocabulary rather than inventing a department-based or
    per-project scheme.
  - **Ingest-side authorization** — deferred at the time this note was
    written; `POST /documents/ingest` now requires authentication (see
    "Post-Phase-11 work" below), though it still doesn't check whether the
    caller's clearance justifies the `access_level` they're requesting for
    the upload — that's a separate, more nuanced policy question left
    open.
  - **`Document.department` (string) vs. `Employee.department_id` (FK)**
    reconciliation — still two independent, unreconciled representations
    of "department." Deferred again.
- `/query`'s generated answer only covers the document-RAG and graph-
  employee-search evidence paths. `comparison` and `cv_generation` intents
  route to the same graph queries as `employee_search` (per
  `plan_query()`) and `/query` gives a generic grounded answer over those
  matches, not anything tailored to "compare" or "generate a CV" — CV
  generation now has its own dedicated pipeline/endpoints (Phase 10,
  `/employees/{id}/cv` and `/employees/bulk-cv`, not `/query`), but
  multi-employee comparison synthesis still has no dedicated handling.
- Confidence values ("evidence_supported" / "insufficient_evidence" /
  "conflicting_evidence") are the LLM's own self-assessment, not a
  calibrated probability or independently verified score — spec section
  25 explicitly warns against representing heuristic confidence as a
  calibrated probability, and this implementation doesn't. Independent
  faithfulness/citation-correctness scoring against a labeled benchmark is
  Phase 11 (evaluation).
- Phase 10: `/employees/*` (profile fetch, single CV, bulk CV) require
  authentication as a baseline but are NOT clearance-differentiated — same
  documented gap as `/graph/*` since Phase 6/8 (the `Employee` entity has
  no classification field). A CV can currently be generated for any
  employee by any authenticated user regardless of clearance level; worth
  revisiting once/if employee data gets its own access-level field.
- CV generation intentionally never invokes the LLM (see
  `app/services/documents/cv_generator.py`'s module docstring) — this is a
  deliberate design choice trading a more "polished" prose summary for a
  hard guarantee against invented experience, not an oversight. If a
  future iteration wants an LLM-written summary section, it would need
  the same evidence-grounding rigor as Phase 9's answer generation, not a
  simpler treatment just because it's "only" a CV.
- **Phase 11 evaluation benchmark is intentionally small and
  high-certainty-ground-truth**, not a stress test: 16 cases, all
  confidently answerable from data that's actually in the system. It
  doesn't yet exercise ambiguous phrasing, adversarial/injection content
  in evidence (covered by Phase 9's dedicated injection tests, not by this
  benchmark), or the conflicting-evidence/near-duplicate/restricted-
  document scenarios — those now have real seeded content (see
  "Post-Phase-11 work" above) but haven't been added as
  `evaluation/benchmark.jsonl` cases yet, since `answer_correctness()`'s
  substring-match approach doesn't naturally fit a "did it correctly
  report both conflicting values" check. A 100% accuracy / 1.0 retrieval
  result on this benchmark should be read as "the implemented paths are
  correct on known-answerable questions," not as a general quality claim.
- `answer_correctness()` (`app/services/evaluation/generation_metrics.py`)
  is a case-insensitive substring check against `expected_answer_contains`
  — deliberately simple and dependency-free (no second model, no
  similarity threshold to tune), but it means a correct-but-differently-
  worded answer (e.g. "every 3 months" instead of "quarter") would score
  as incorrect. `--judge` (LLM-as-judge) is the intended complement for
  that gap, not a fix for this metric itself.
- LLM-as-judge (`app/services/evaluation/llm_judge.py`) still uses
  `settings.openai_judge_model`, which defaults to the SAME model as
  generation (`qwen2.5:7b-instruct-q4_K_M` — the only general-purpose
  local model available in this environment; `qwen2.5-coder:14b` is
  code-specialized). This is a real, unmitigated self-preference-bias
  risk, flagged since Phase 7 and still open at the end of Phase 11 — the
  calibration workflow (`scripts/calibrate_judge.py`) exists specifically
  so this can be checked empirically against a human rather than assumed
  away, but hasn't been run against a human rater yet (would need a
  person, not something to fake). Revisit if a second, genuinely
  different general-purpose local model becomes available.
- The evaluation fixture documents (`evaluation/fixtures/docs/*.txt`) are
  still ingested and deleted within a single `scripts/evaluate.py` run
  rather than kept in the permanent demo corpus — this was necessary at
  the time because of an existing test's empty-corpus assumption (since
  fixed, see "Post-Phase-11 work" above), and is kept as good hygiene
  regardless (an eval run's fixtures are run-scoped data, not demo
  corpus content, same reasoning as any other test fixture in this
  project).

## Next tasks

1. Entity extraction from ingested document text into the graph (spec
   section 6's ingestion pipeline step, flagged as an unscheduled gap
   since Phase 6).
2. A custom UI — **not Streamlit** (explicit user direction, overriding
   the `ui` extra's original plan; `pyproject.toml`'s `ui` extra needs
   updating to drop the `streamlit` dependency accordingly).
3. Human calibration of the LLM judge (`python scripts/calibrate_judge.py`)
   against a real human rater, now that `scripts/evaluate.py --judge` can
   actually produce `evaluation/judge_results.jsonl` to calibrate against
   — needs a person, not something that can be done autonomously.
