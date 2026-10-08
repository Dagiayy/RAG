"""Query understanding: natural-language question -> validated QueryIntent.

The LLM proposes a candidate JSON object (constrained at decode time via
the OpenAI-compatible `response_format={"type": "json_schema", ...}`
structured-output parameter — see ADR 0005); Pydantic then validates it.
On validation failure, the error is fed back to the model for one retry;
if that also fails, we fall back to `general_qa` (routes to plain hybrid
document retrieval, see app/pipelines/query_planner.py) rather than crash
the request — an LLM being unavailable or unreliable must degrade
gracefully (spec section 36 failure testing), not break the query endpoint
outright. The model's raw output is JSON data, never executed directly.
"""

import json

import structlog
from pydantic import ValidationError

from app.config import get_settings
from app.schemas.query_intent import QueryIntent
from app.services.generation.llm_client import get_llm_client

logger = structlog.get_logger("query_understanding")

SYSTEM_PROMPT = """You convert a user's natural-language enterprise question into structured JSON.

Valid "intent" values:
- employee_search: asking which employees have some skill/certification/experience
- project_lookup: asking about a specific named project (e.g. who worked on it)
- document_qa: asking about policy/procedure/document content, not people or projects
- cv_generation: asking to generate a CV/resume for matching employees
- comparison: asking to compare multiple employees
- general_qa: anything that doesn't clearly fit the above

"entities" may include these keys, only when clearly present in the question:
- skill (e.g. "GIS Mapping", "LoRaWAN/IoT Sensors")
- industry (e.g. "Agriculture", "Humanitarian Aid")
- certification (e.g. "Certified M&E Professional")
- client (e.g. "AgriRise Foundation")
- project (an exact project name)
- role (a job title)

"filters" is a list of {"field": "years_experience", "operator": one of ">" ">=" "<" "<=" "==" "!=", "value": number} — only include when the question mentions years of experience.

"requested_output" is a short free-text label for what the user wants back, e.g. "employee_list", "answer", "cv".

Respond with JSON only, matching the schema. Do not invent entity values that aren't stated or clearly implied in the question.

Examples:
Q: "Which employees have GIS Mapping experience?"
A: {"intent": "employee_search", "entities": {"skill": "GIS Mapping"}, "filters": [], "requested_output": "employee_list"}

Q: "Which employees with Survey Design skills worked on Humanitarian Aid projects with more than 5 years experience?"
A: {"intent": "employee_search", "entities": {"skill": "Survey Design", "industry": "Humanitarian Aid"}, "filters": [{"field": "years_experience", "operator": ">", "value": 5}], "requested_output": "employee_list"}

Q: "Who worked on the Refugee Camp Needs Assessment project?"
A: {"intent": "project_lookup", "entities": {"project": "Refugee Camp Needs Assessment"}, "filters": [], "requested_output": "employee_list"}

Q: "What does the confined space entry safety procedure require?"
A: {"intent": "document_qa", "entities": {}, "filters": [], "requested_output": "answer"}
"""


async def understand_query(query: str, max_retries: int = 1) -> QueryIntent:
    settings = get_settings()
    client = get_llm_client()
    schema = QueryIntent.model_json_schema()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "QueryIntent", "schema": schema, "strict": True},
                },
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return QueryIntent.model_validate(data)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("query_understanding_invalid_output", attempt=attempt, error=str(exc))
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"That response was invalid: {exc}. Respond again with valid JSON matching the schema.",
                }
            )
        except Exception as exc:  # LLM unreachable/unavailable — degrade, don't crash
            last_error = exc
            logger.warning("query_understanding_llm_error", attempt=attempt, error=str(exc))
            break

    logger.warning("query_understanding_fallback_to_general_qa", error=str(last_error))
    return QueryIntent(intent="general_qa", requested_output="answer")
