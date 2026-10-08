"""Human calibration of the LLM judge (docs/evaluation.md): measures
agreement between judge scores and a human's own scores for a sample of
judged cases, so judge output can be reported as calibrated (or not)
rather than blindly trusted.

Quadratic-weighted Cohen's kappa is used rather than plain agreement rate
because the judge/human scores here are ordinal (1-5): a judge/human
mismatch of 1 vs 5 is a much worse disagreement than 3 vs 4, and weighted
kappa accounts for that while plain percent-agreement would not. The
weighting assumes contiguous integer score levels (true for every 1-5
rubric score produced by `llm_judge.py`).
"""

from dataclasses import dataclass


@dataclass
class CalibrationResult:
    n: int
    agreement_rate: float
    weighted_kappa: float


def weighted_cohens_kappa(
    judge_scores: list[int], human_scores: list[int], levels: list[int] | None = None
) -> float:
    """Quadratic-weighted Cohen's kappa. 1.0 = perfect agreement, 0.0 =
    chance-level agreement, negative = worse than chance."""
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must be the same length")
    if not judge_scores:
        raise ValueError("score lists must be non-empty")

    levels = levels or sorted(set(judge_scores) | set(human_scores))
    n = len(judge_scores)
    span = (max(levels) - min(levels)) or 1

    def weight(a: int, b: int) -> float:
        return ((a - b) / span) ** 2

    observed = {(j, h): 0 for j in levels for h in levels}
    for j, h in zip(judge_scores, human_scores, strict=True):
        observed[(j, h)] += 1

    judge_marginal = {lvl: sum(1 for j in judge_scores if j == lvl) for lvl in levels}
    human_marginal = {lvl: sum(1 for h in human_scores if h == lvl) for lvl in levels}

    numerator = sum(weight(j, h) * observed[(j, h)] for j in levels for h in levels)
    denominator = sum(
        weight(j, h) * judge_marginal[j] * human_marginal[h] / n for j in levels for h in levels
    )
    if denominator == 0:
        return 1.0  # no variance in either rater — trivially perfect agreement
    return 1 - numerator / denominator


def calibrate(judge_scores: list[int], human_scores: list[int]) -> CalibrationResult:
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must be the same length")
    if not judge_scores:
        raise ValueError("score lists must be non-empty")
    n = len(judge_scores)
    agreement_rate = sum(1 for j, h in zip(judge_scores, human_scores, strict=True) if j == h) / n
    kappa = weighted_cohens_kappa(judge_scores, human_scores)
    return CalibrationResult(n=n, agreement_rate=agreement_rate, weighted_kappa=kappa)
