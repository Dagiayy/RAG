# Evaluation Architecture

## Benchmark dataset

`evaluation/benchmark.jsonl` — one JSON object per case:

```json
{
  "id": "q001",
  "question": "Which employees have Siemens PLC experience?",
  "expected_answer_contains": ["..."],
  "expected_source_document_ids": ["..."],
  "expected_entities": {"skill": "Siemens PLC"},
  "expected_filters": [],
  "authorized_user": "manager_ops"
}
```

Generated alongside the synthetic dataset (Phase 9/synthetic data step) so
expected documents/entities are known ground truth, not guessed.

## Retrieval evaluation (independent of generation)

`app/services/evaluation/retrieval_metrics.py` computes, per query, against
`expected_source_document_ids`:

- Recall@K, Precision@K, MRR, Hit Rate, NDCG@K

Run separately from generation so a bad prompt doesn't mask good retrieval
or vice versa.

## Generation evaluation

`app/services/evaluation/generation_metrics.py`:

- **Answer correctness** — string/semantic match against
  `expected_answer_contains`.
- **Faithfulness** — every claim in the answer must be traceable to a cited
  chunk (checked via the same citation objects the API returns, not
  re-parsed from prose).
- **Context relevance** — fraction of retrieved evidence actually used.
- **Citation correctness/completeness** — cited doc/chunk ids exist in the
  evidence set and cover the claims made.

## LLM-as-judge

`app/services/evaluation/llm_judge.py` — a separate Ollama call (can use a
different/larger local model than the answering model to reduce
self-preference bias) scores correctness/relevance/groundedness/citation
quality on a 1-5 scale with a rubric prompt. Judge output is stored, never
silently trusted:

- `evaluation/judge_results.jsonl`: question, model answer, judge score,
  judge rationale.
- Human calibration workflow: `app/services/evaluation/calibration.py` lets
  a human enter a score for a sample of judged cases; agreement (e.g.
  weighted Cohen's kappa) between human and judge scores is computed and
  reported in `docs/implementation-status.md`'s evaluation section. Judge
  scores are treated as a heuristic signal, not ground truth, in any
  reported metric.

## System metrics

Logged per request from observability data: latency (per stage: vector,
BM25, graph, SQL, rerank, LLM, total), throughput under load test, failure
rate, retrieval candidate counts, token usage.

## Running evaluation

`scripts/evaluate.py` (invoked by `make evaluate` / equivalent) runs the
full benchmark against a running instance, writes a report to
`evaluation/reports/<timestamp>.json`, and prints a summary table. CI runs a
small smoke subset on PRs; the full benchmark is run manually/scheduled.
