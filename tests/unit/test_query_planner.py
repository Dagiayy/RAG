from app.pipelines.query_planner import describe_graph_query, plan_query
from app.schemas.query_intent import QueryIntent


def test_employee_search_with_skill_and_industry_routes_to_multi_hop():
    intent = QueryIntent(
        intent="employee_search", entities={"skill": "GIS Mapping", "industry": "Agriculture"}
    )
    plan = plan_query(intent)
    assert plan.use_graph is True
    assert plan.graph_query_type == "by_skill_and_industry"
    assert plan.use_document_rag is False


def test_employee_search_with_skill_only_routes_to_skill_query():
    intent = QueryIntent(intent="employee_search", entities={"skill": "GIS Mapping"})
    plan = plan_query(intent)
    assert plan.graph_query_type == "by_skill"


def test_employee_search_with_industry_only_routes_to_industry_query():
    intent = QueryIntent(intent="employee_search", entities={"industry": "Agriculture"})
    plan = plan_query(intent)
    assert plan.graph_query_type == "by_industry"


def test_employee_search_with_certification_routes_to_certification_query():
    intent = QueryIntent(
        intent="employee_search", entities={"certification": "Certified M&E Professional"}
    )
    plan = plan_query(intent)
    assert plan.graph_query_type == "by_certification"


def test_employee_search_with_client_routes_to_client_query():
    intent = QueryIntent(intent="employee_search", entities={"client": "AgriRise Foundation"})
    plan = plan_query(intent)
    assert plan.graph_query_type == "by_client"


def test_employee_search_prefers_skill_and_industry_over_certification():
    intent = QueryIntent(
        intent="employee_search",
        entities={"skill": "GIS Mapping", "industry": "Agriculture", "certification": "X"},
    )
    plan = plan_query(intent)
    assert plan.graph_query_type == "by_skill_and_industry"


def test_employee_search_with_no_graph_entities_falls_back_to_document_rag():
    intent = QueryIntent(intent="employee_search", entities={})
    plan = plan_query(intent)
    assert plan.use_graph is False
    assert plan.use_document_rag is True


def test_project_lookup_with_project_entity_routes_to_project_team():
    intent = QueryIntent(
        intent="project_lookup", entities={"project": "Refugee Camp Needs Assessment"}
    )
    plan = plan_query(intent)
    assert plan.graph_query_type == "project_team"
    assert plan.use_document_rag is False


def test_project_lookup_without_project_entity_falls_back_to_document_rag():
    intent = QueryIntent(intent="project_lookup", entities={})
    plan = plan_query(intent)
    assert plan.use_graph is False
    assert plan.use_document_rag is True


def test_document_qa_routes_to_document_rag_only():
    intent = QueryIntent(intent="document_qa", entities={})
    plan = plan_query(intent)
    assert plan.use_graph is False
    assert plan.use_document_rag is True


def test_general_qa_routes_to_document_rag_only():
    intent = QueryIntent(intent="general_qa", entities={})
    plan = plan_query(intent)
    assert plan.use_graph is False
    assert plan.use_document_rag is True


def test_comparison_intent_with_skill_routes_to_graph():
    intent = QueryIntent(intent="comparison", entities={"skill": "GIS Mapping"})
    plan = plan_query(intent)
    assert plan.use_graph is True
    assert plan.graph_query_type == "by_skill"


def test_cv_generation_with_skill_routes_to_graph():
    intent = QueryIntent(intent="cv_generation", entities={"skill": "GIS Mapping"})
    plan = plan_query(intent)
    assert plan.use_graph is True
    assert plan.graph_query_type == "by_skill"


def test_describe_graph_query_by_skill():
    intent = QueryIntent(intent="employee_search", entities={"skill": "GIS Mapping"})
    plan = plan_query(intent)
    assert describe_graph_query(plan, intent) == "employees with skill 'GIS Mapping'"


def test_describe_graph_query_by_skill_and_industry():
    intent = QueryIntent(
        intent="employee_search", entities={"skill": "GIS Mapping", "industry": "Agriculture"}
    )
    plan = plan_query(intent)
    description = describe_graph_query(plan, intent)
    assert "GIS Mapping" in description
    assert "Agriculture" in description


def test_describe_graph_query_project_team():
    intent = QueryIntent(intent="project_lookup", entities={"project": "Drought Monitoring"})
    plan = plan_query(intent)
    assert (
        describe_graph_query(plan, intent) == "employees who worked on project 'Drought Monitoring'"
    )


def test_describe_graph_query_returns_none_when_no_graph_query():
    intent = QueryIntent(intent="document_qa", entities={})
    plan = plan_query(intent)
    assert describe_graph_query(plan, intent) is None
