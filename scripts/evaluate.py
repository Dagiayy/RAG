"""Phase 11 evaluation: runs evaluation/benchmark.jsonl against the live
query pipeline (Postgres + Qdrant + Neo4j + Ollama all required — the same
stack `run_query()` always needs), computes retrieval + generation metrics
per docs/evaluation.md, optionally scores each answer with the LLM judge,
and writes a timestamped report to evaluation/reports/.

Document-RAG benchmark cases need real, known documents to check retrieval
against. This script ingests evaluation/fixtures/docs/*.txt at the start of
the run and deletes them (Postgres row + Qdrant vectors + BM25 index
invalidation) at the end — the same track-then-clean-up pattern this
project's integration tests use — so a benchmark run never permanently
pollutes the demo corpus or changes the behavior of tests/manual queries
that assume an empty document corpus (e.g.
tests/integration/test_query_pipeline.py's "no documents ingested yet"
case).

Graph-routed cases need no fixture setup: they're scored against
scripts/seed_demo_data.py's already-seeded employees by employee_code, so
run that script first if the database is empty.

Run: python scripts/evaluate.py [--judge] [--k N]
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.models.document import Document
from app.models.enterprise import Employee
from app.pipelines.query_pipeline import run_query
from app.repositories.db import get_session_factory
from app.services.evaluation.generation_metrics import (
    answer_correctness,
    citation_precision,
    citation_recall,
    context_precision,
)
from app.services.evaluation.llm_judge import judge_answer
from app.services.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    average_metrics,
    compute_retrieval_metrics,
)
from app.services.ingestion.hashing import sha256_bytes
from app.services.ingestion.pipeline import IngestionError, ingest_document
from app.services.retrieval.bm25 import invalidate_bm25_index
from app.services.retrieval.vector import delete_document_vectors

BENCHMARK_PATH = Path("evaluation/benchmark.jsonl")
FIXTURES_DIR = Path("evaluation/fixtures/docs")
REPORTS_DIR = Path("evaluation/reports")
JUDGE_RESULTS_PATH = Path("evaluation/judge_results.jsonl")

DEFAULT_K = 10


def _load_benchmark() -> list[dict]:
    cases = []
    with BENCHMARK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


async def _ensure_fixtures_ingested(session) -> dict[str, str]:
    """Ingests every evaluation/fixtures/docs/*.txt not already present
    (matched by checksum, the same dedup key ingest_document() already
    enforces). Returns {filename: document_id}."""
    filename_to_id: dict[str, str] = {}
    ingested_this_run: list[str] = []

    for path in sorted(FIXTURES_DIR.glob("*.txt")):
        content = path.read_bytes()
        checksum = sha256_bytes(content)
        existing = await session.scalar(select(Document).where(Document.checksum == checksum))
        if existing is not None:
            filename_to_id[path.name] = str(existing.id)
            continue
        try:
            result = await ingest_document(session, path.name, content, title=path.stem)
        except IngestionError as exc:
            print(f"  WARNING: failed to ingest fixture {path.name}: {exc}", file=sys.stderr)
            continue
        filename_to_id[path.name] = str(result.document_id)
        ingested_this_run.append(path.name)

    if ingested_this_run:
        print(
            f"Ingested {len(ingested_this_run)} evaluation fixture document(s): "
            f"{ingested_this_run}"
        )
    return filename_to_id


async def _cleanup_fixtures(session, filename_to_id: dict[str, str]) -> None:
    for document_id in filename_to_id.values():
        doc = await session.get(Document, uuid.UUID(document_id))
        if doc is not None:
            delete_document_vectors(doc.id)
            await session.delete(doc)
    await session.commit()
    invalidate_bm25_index()
    print(f"Cleaned up {len(filename_to_id)} evaluation fixture document(s).")


async def _employee_codes_to_pg_ids(session, codes: list[str]) -> set[str]:
    rows = await session.execute(select(Employee.id).where(Employee.employee_code.in_(codes)))
    return {str(row[0]) for row in rows}


async def run_evaluation(k: int, run_judge: bool) -> dict:
    cases = _load_benchmark()
    factory = get_session_factory()

    async with factory() as session:
        filename_to_id = await _ensure_fixtures_ingested(session)

    retrieval_metrics_by_case: list[RetrievalMetrics] = []
    case_reports: list[dict] = []
    judge_rows: list[dict] = []

    try:
        async with factory() as session:
            for case in cases:
                result = await run_query(session, case["question"], evidence_top_k=max(k, 8))

                if case["type"] == "graph":
                    relevant_ids = await _employee_codes_to_pg_ids(
                        session, case["expected_employee_codes"]
                    )
                    retrieved_ids = [m.pg_id for m in result.graph_matches]
                    cited_ids = [
                        c.employee_pg_id for c in result.generated.citations if c.employee_pg_id
                    ]
                else:
                    relevant_ids = {
                        filename_to_id[fn]
                        for fn in case["expected_source_filenames"]
                        if fn in filename_to_id
                    }
                    retrieved_ids = [r.document_id for r in result.document_evidence]
                    cited_ids = [c.document_id for c in result.generated.citations if c.document_id]

                if not relevant_ids:
                    print(
                        f"  WARNING: case {case['id']} has no resolvable ground truth "
                        "(fixture ingestion may have failed) — skipping.",
                        file=sys.stderr,
                    )
                    continue

                retrieval = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
                retrieval_metrics_by_case.append(retrieval)

                correct = answer_correctness(
                    result.generated.answer, case["expected_answer_contains"]
                )
                case_report = {
                    "id": case["id"],
                    "type": case["type"],
                    "question": case["question"],
                    "answer": result.generated.answer,
                    "confidence": result.generated.confidence,
                    "answer_correct": correct,
                    "retrieval": vars(retrieval),
                    "citation_precision": citation_precision(cited_ids, relevant_ids),
                    "citation_recall": citation_recall(cited_ids, relevant_ids),
                    "context_precision": context_precision(retrieved_ids, relevant_ids),
                }

                if run_judge:
                    judge_result = await judge_answer(
                        case["id"],
                        case["question"],
                        case["expected_answer_contains"],
                        result.generated.answer,
                    )
                    if judge_result is not None:
                        case_report["judge"] = vars(judge_result)
                        judge_rows.append(
                            {
                                "case_id": case["id"],
                                "question": case["question"],
                                "answer": result.generated.answer,
                                "correctness": judge_result.correctness,
                                "relevance": judge_result.relevance,
                                "groundedness": judge_result.groundedness,
                                "citation_quality": judge_result.citation_quality,
                                "rationale": judge_result.rationale,
                            }
                        )

                case_reports.append(case_report)
                print(
                    f"  [{case['id']}] correct={correct} recall@{k}={retrieval.recall_at_k:.2f} "
                    f"mrr={retrieval.mrr:.2f}"
                )
    finally:
        async with factory() as session:
            await _cleanup_fixtures(session, filename_to_id)

    overall_retrieval = (
        average_metrics(retrieval_metrics_by_case) if retrieval_metrics_by_case else None
    )
    accuracy = (
        sum(1 for c in case_reports if c["answer_correct"]) / len(case_reports)
        if case_reports
        else 0.0
    )

    if judge_rows:
        with JUDGE_RESULTS_PATH.open("w", encoding="utf-8") as f:
            for row in judge_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(judge_rows)} judge result(s) to {JUDGE_RESULTS_PATH}")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "k": k,
        "num_cases": len(cases),
        "num_scored": len(case_reports),
        "answer_accuracy": accuracy,
        "retrieval": vars(overall_retrieval) if overall_retrieval else None,
        "judge_coverage": f"{len(judge_rows)}/{len(case_reports)}" if run_judge else "not run",
        "cases": case_reports,
    }


def _print_summary(report: dict) -> None:
    print("\n=== Evaluation summary ===")
    print(f"Cases: {report['num_scored']}/{report['num_cases']} scored")
    print(f"Answer accuracy (substring match): {report['answer_accuracy']:.1%}")
    if report["retrieval"]:
        r = report["retrieval"]
        print(
            f"Retrieval @ k={report['k']}: recall={r['recall_at_k']:.2f} "
            f"precision={r['precision_at_k']:.2f} mrr={r['mrr']:.2f} "
            f"hit_rate={r['hit_rate']:.2f} ndcg={r['ndcg_at_k']:.2f}"
        )
    print(f"LLM judge coverage: {report['judge_coverage']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation benchmark.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Retrieval cutoff (default 10).")
    parser.add_argument(
        "--judge", action="store_true", help="Also score answers with the LLM judge."
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run_evaluation(args.k, args.judge))
    _print_summary(report)

    report_path = REPORTS_DIR / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
