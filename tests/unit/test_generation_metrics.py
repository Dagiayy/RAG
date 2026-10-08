import pytest

from app.services.evaluation.generation_metrics import (
    answer_correctness,
    citation_precision,
    citation_recall,
    context_precision,
)


def test_answer_correctness_true_when_all_fragments_present():
    assert answer_correctness("The remote work limit is 3 days per week.", ["3 days"])


def test_answer_correctness_false_when_a_fragment_missing():
    assert not answer_correctness("I could not find sufficient evidence.", ["3 days"])


def test_answer_correctness_case_insensitive():
    assert answer_correctness("THE LIMIT IS 3 DAYS", ["3 days"])


def test_answer_correctness_no_expected_fragments_is_vacuously_true():
    assert answer_correctness("anything at all", [])


def test_context_precision_fraction_relevant():
    assert context_precision(["a", "b", "x", "y"], {"a", "b"}) == 0.5


def test_context_precision_no_evidence_is_zero():
    assert context_precision([], {"a"}) == 0.0


def test_context_precision_all_relevant_is_1():
    assert context_precision(["a", "b"], {"a", "b", "c"}) == 1.0


def test_citation_precision_fraction_of_cites_that_are_relevant():
    assert citation_precision(["a", "x"], {"a", "b"}) == 0.5


def test_citation_precision_nothing_cited_is_zero():
    assert citation_precision([], {"a"}) == 0.0


def test_citation_recall_fraction_of_relevant_that_got_cited():
    assert citation_recall(["a"], {"a", "b"}) == 0.5


def test_citation_recall_full_coverage():
    assert citation_recall(["a", "b", "x"], {"a", "b"}) == 1.0


def test_citation_recall_empty_relevant_raises():
    with pytest.raises(ValueError):
        citation_recall(["a"], set())
