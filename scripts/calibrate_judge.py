"""Human calibration workflow for the LLM judge (docs/evaluation.md).

Reads evaluation/judge_results.jsonl (written by `scripts/evaluate.py
--judge`), shows a human rater the question + answer for a sample of
judged cases (the judge's own score/rationale is hidden until after the
human scores it, to avoid anchoring), collects a human "correctness"
score for each, and reports agreement (plain rate + quadratic-weighted
Cohen's kappa) between the judge and the human on that dimension.

Interactive only — not run in CI or any automated test; the underlying
`calibrate()`/`weighted_cohens_kappa()` math is covered by
tests/unit/test_calibration.py.

Run: python scripts/calibrate_judge.py [--sample N]
"""

import argparse
import json
from pathlib import Path

from app.services.evaluation.calibration import calibrate

JUDGE_RESULTS_PATH = Path("evaluation/judge_results.jsonl")


def _load_judge_results() -> list[dict]:
    if not JUDGE_RESULTS_PATH.exists():
        raise SystemExit(
            f"{JUDGE_RESULTS_PATH} not found — run `python scripts/evaluate.py --judge` first."
        )
    rows = []
    with JUDGE_RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_human_score(question: str, answer: str) -> int:
    print("\n" + "-" * 60)
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
    while True:
        raw = input("Your correctness score (1-5): ").strip()
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Please enter an integer from 1 to 5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against a human rater.")
    parser.add_argument(
        "--sample", type=int, default=10, help="Number of cases to rate (default 10)."
    )
    args = parser.parse_args()

    rows = _load_judge_results()
    sample = rows[: args.sample]

    print(f"Rating {len(sample)} of {len(rows)} judged case(s). Score 'correctness' only (1-5).")

    human_scores = []
    judge_scores = []
    for row in sample:
        human_scores.append(_prompt_human_score(row["question"], row["answer"]))
        judge_scores.append(row["correctness"])

    result = calibrate(judge_scores, human_scores)

    print("\n=== Calibration report ===")
    print(f"n = {result.n}")
    print(f"Agreement rate: {result.agreement_rate:.1%}")
    print(f"Weighted Cohen's kappa: {result.weighted_kappa:.3f}")
    print(
        "(kappa: >0.8 near-perfect, 0.6-0.8 substantial, 0.4-0.6 moderate, "
        "<0.4 judge scores should not be trusted without human review)"
    )


if __name__ == "__main__":
    main()
