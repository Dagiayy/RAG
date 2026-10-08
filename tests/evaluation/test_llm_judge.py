"""Requires Ollama running (self-skips otherwise, same pattern as
tests/integration/*). Qualitative checks only — an LLM judge's exact
scores aren't deterministic enough to assert precise values against, but a
clearly-correct answer should consistently outscore a clearly-wrong one on
correctness, which is the property the whole judge mechanism depends on.
"""

import pytest

from app.services.evaluation.llm_judge import judge_answer
from tests.conftest import ollama_is_reachable

pytestmark = pytest.mark.skipif(not ollama_is_reachable(), reason="Ollama not reachable")


@pytest.mark.asyncio
async def test_judge_scores_correct_answer_higher_than_wrong_answer():
    question = "How many days per week can non-field employees work remotely?"
    expected = ["3 days"]

    good = await judge_answer(
        "calibration-good",
        question,
        expected,
        "Non-field employees may work remotely up to 3 days per week, "
        "subject to manager approval [1].",
    )
    bad = await judge_answer(
        "calibration-bad",
        question,
        expected,
        "Employees may work remotely every day of the week with no restrictions.",
    )

    assert good is not None
    assert bad is not None
    assert good.correctness > bad.correctness
    assert 1 <= good.correctness <= 5
    assert 1 <= bad.correctness <= 5


@pytest.mark.asyncio
async def test_judge_degrades_to_none_on_unreachable_model(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "openai_judge_model", "this-model-does-not-exist")

    result = await judge_answer("missing-model", "What is X?", ["X"], "X is Y.")
    assert result is None
