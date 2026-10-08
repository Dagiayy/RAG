# Retrieval Design

## Goals

- No single retriever is trusted alone. Vector search misses exact
  identifiers (employee IDs, model numbers); BM25 misses paraphrase; SQL/graph
  miss free-text nuance; each covers the others' blind spot.
- Query planning avoids unnecessary retrieval (a lookup question shouldn't
  trigger a full RAG pipeline).
- Every answer traces to retrieved evidence with real citations.

## Query understanding → structured intent

LLM-assisted extraction into a **validated Pydantic schema**
(`app/schemas/query_intent.py`):

```python
class QueryIntent(BaseModel):
    intent: Literal["employee_search", "document_qa", "project_lookup",
                     "cv_generation", "comparison", "general_qa"]
    entities: dict[str, str] = {}
    filters: list[QueryFilter] = []
    requested_output: str
```

The LLM proposes a candidate; Pydantic validates it; on validation failure we
retry once with the error fed back, then fall back to `general_qa` (full RAG,
no structured filters) rather than crash. The LLM's output is never executed
directly — it only parameterizes typed retriever calls.

## Query planning

`app/pipelines/query_planner.py` maps `QueryIntent` → a set of retrievers to
run:

| Intent shape | Retrievers |
|---|---|
| Pure lookup (e.g. "who is PM for Project Alpha") | SQL and/or graph only |
| Document content question | Vector + BM25 + rerank |
| Entity search with structured filters (role/skill/experience) | SQL + graph, vector/BM25 only if free-text criteria present |
| Multi-hop entity + document question | SQL + graph + vector + BM25, fused |

Planner never runs a retriever whose required entities/filters are absent
from the parsed intent.

## Hybrid fusion

Vector (cosine similarity, Qdrant) and BM25 (rank-bm25) candidate lists are
combined with **Reciprocal Rank Fusion**:

```
score(d) = sum over retrievers r containing d of  1 / (k + rank_r(d))
```

`k` (default 60) and per-retriever weights are configurable
(`configs/retrieval.yaml`). RRF is chosen over raw score normalization
because BM25 and cosine scores are not on comparable scales and
min-max normalization is sensitive to outliers; RRF only needs rank order.
Alternative fusion strategies are pluggable behind a `FusionStrategy`
interface for future experimentation (weighted sum, learned fusion).

## Reranking

Top ~30 fused candidates go through a CrossEncoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2` default, configurable) which scores
`(query, chunk)` pairs jointly — much more accurate than bi-encoder cosine
similarity because it attends across both texts, at the cost of being too
slow to run over the full corpus. That's why it's a second stage over a
small candidate set, not the primary retriever. Output: top 5–10 evidence
chunks (configurable).

## Graph retrieval

Cypher templates parameterized from `QueryIntent.entities`/`filters`
(`app/services/graph/queries.py`), e.g. skill/industry/experience traversal.
No free-text-to-Cypher LLM generation in v1 — arbitrary LLM-generated Cypher
against a real graph is an injection and correctness risk; templates cover
the documented query shapes (section 13/17 of the spec) and are extended as
new question shapes are needed.

## Evidence assembly, duplicates, conflicts

- **Duplicate/redundancy filter**: exact (checksum) and near-duplicate
  (embedding cosine > threshold) chunks are collapsed to one representative
  with all corroborating sources retained in `evidence.corroborated_by`.
- **Conflict detection**: chunks answering the same normalized claim
  (same entity/attribute pair) with differing values are flagged, not
  averaged; the response surfaces both with source/version/date and lets
  metadata rules (effective date, `document_status=active`, supersedes chain)
  pick the authoritative one where the rules are unambiguous, otherwise the
  system reports the conflict to the user instead of picking.

## Context construction & grounding

Selected evidence (chunks + graph facts + SQL rows) is serialized into a
structured context block with explicit source tags. The generation prompt
instructs the model to answer only from that block, treat document text as
data not instructions (see `docs/security.md`), cite every claim, and emit
an explicit insufficient-evidence response when nothing clears the
similarity/rerank-score threshold (`configs/retrieval.yaml:
min_evidence_score`).

## Evaluation hooks

Every retrieval call logs candidates, scores, and the retrievers used per
stage, so `app/services/evaluation` can compute Recall@K/MRR/NDCG against
`evaluation/benchmark.jsonl` independent of generation quality.
