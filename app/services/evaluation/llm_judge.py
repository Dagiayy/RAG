"""LLM-as-judge (docs/evaluation.md): a second Ollama call (via the same
OpenAI-compatible client as answer generation — ADR 0005) scores a
generated answer against the question + expected facts on a 1-5 rubric
across four dimensions.

Uses `settings.openai_judge_model` — currently the SAME model as
generation (`qwen2.5:7b-instruct-q4_K_M`, the only model available in this
environment), a known self-preference-bias risk flagged since Phase 7. Per
docs/evaluation.md, judge scores are a heuristic signal to be calibrated
against human scores (see `calibration.py`), never trusted as ground
truth on their own.
"""

import json
from dataclasses import dataclass

import structlog
from pydantic import BaseModel

from app.config import get_settings
from app.services.generation.llm_client import get_llm_client

logger = structlog.get_logger("llm_judge")

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring a RAG system's answer to a question.

Score the ANSWER on four dimensions, each an integer 1-5 (5 = best):
- correctness: does the answer match the expected facts?
- relevance: does the answer actually address the question asked?
- groundedness: does the answer stick to the provided evidence rather than inventing facts?
- citation_quality: are the citations appropriate, not missing or extraneous?

Be strict: a vague, partially-wrong, or unsupported answer should NOT score 5.

Respond with JSON matching the schema: correctness (int), relevance (int),
groundedness (int), citation_quality (int), rationale (string, one or two
sentences)."""


class _JudgeOutput(BaseModel):
    correctness: int
    relevance: int
    groundedness: int
    citation_quality: int
    rationale: str


@dataclass
class JudgeResult:
    case_id: str
    correctness: int
    relevance: int
    groundedness: int
    citation_quality: int
    rationale: str


async def judge_answer(
    case_id: str,
    question: str,
    expected_answer_contains: list[str],
    generated_answer: str,
) -> JudgeResult | None:
    """Returns None (rather than raising) if the judge call fails or
    degrades — a flaky/unreachable judge shouldn't crash an otherwise-valid
    evaluation run; callers report judge coverage separately from the
    deterministic benchmark metrics."""
    settings = get_settings()
    client = get_llm_client()
    schema = _JudgeOutput.model_json_schema()

    user_message = (
        f"Question: {question}\n"
        f"Expected facts the answer should contain: {expected_answer_contains}\n"
        f"Answer given by the system under test: {generated_answer}"
    )

    try:
        response = await client.chat.completions.create(
            model=settings.openai_judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "JudgeOutput", "schema": schema, "strict": True},
            },
        )
        data = json.loads(response.choices[0].message.content)
        parsed = _JudgeOutput.model_validate(data)
    except Exception as exc:  # invalid output or LLM unreachable — degrade, don't crash
        logger.warning("llm_judge_failed", case_id=case_id, error=str(exc))
        return None

    return JudgeResult(
        case_id=case_id,
        correctness=parsed.correctness,
        relevance=parsed.relevance,
        groundedness=parsed.groundedness,
        citation_quality=parsed.citation_quality,
        rationale=parsed.rationale,
    )
