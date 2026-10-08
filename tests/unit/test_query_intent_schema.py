import pytest
from pydantic import ValidationError

from app.schemas.query_intent import QueryFilter, QueryIntent


def test_minimal_valid_intent():
    intent = QueryIntent(intent="general_qa")
    assert intent.entities == {}
    assert intent.filters == []
    assert intent.requested_output == "answer"


def test_rejects_unknown_intent_value():
    with pytest.raises(ValidationError):
        QueryIntent(intent="not_a_real_intent")


def test_filter_requires_known_field():
    with pytest.raises(ValidationError):
        QueryFilter(field="salary", operator=">", value=100000)


def test_filter_requires_known_operator():
    with pytest.raises(ValidationError):
        QueryFilter(field="years_experience", operator="~=", value=5)


def test_full_intent_with_entities_and_filters():
    intent = QueryIntent(
        intent="employee_search",
        entities={"skill": "GIS Mapping", "industry": "Agriculture"},
        filters=[{"field": "years_experience", "operator": ">", "value": 5}],
        requested_output="employee_list",
    )
    assert intent.entities["skill"] == "GIS Mapping"
    assert intent.filters[0].operator == ">"


def test_model_json_schema_is_generatable():
    # sanity check that this schema can actually be passed as Ollama's
    # structured-output `format` parameter without erroring
    schema = QueryIntent.model_json_schema()
    assert schema["type"] == "object"
    assert "intent" in schema["properties"]
