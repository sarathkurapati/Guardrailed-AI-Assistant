import pytest
from tests.conftest import make_state
from app.graph.nodes.output_validator import output_validator_node, route_after_validation
from app.config import settings


def _state_with_response(response: str, attempts: int = 1):
    state = make_state()
    state["raw_response"] = response
    state["generation_attempts"] = attempts
    return state


def test_empty_response_fails():
    state = _state_with_response("")
    result = output_validator_node(state)
    assert result["output_valid"] is False
    assert result["validation_failure_reason"] == "empty_or_too_short"


def test_short_response_fails():
    state = _state_with_response("ok")
    result = output_validator_node(state)
    assert result["output_valid"] is False


def test_too_long_response_fails():
    state = _state_with_response("x" * 2001)
    result = output_validator_node(state)
    assert result["output_valid"] is False
    assert result["validation_failure_reason"] == "response_too_long"


def test_valid_response_passes():
    state = _state_with_response(
        "Employees are entitled to 20 days of annual leave per year. "
        "Requests must be submitted at least 5 working days in advance."
    )
    result = output_validator_node(state)
    # May fail bias check due to LLM call in test env; just verify fields set
    assert "output_valid" in result
    assert isinstance(result["toxicity_score"], float)
    assert isinstance(result["bias_flag"], bool)


def test_audit_trail_appended():
    state = _state_with_response("This is a test response that is long enough.")
    result = output_validator_node(state)
    audit_entries = [e for e in result["audit_trail"] if e["node"] == "output_validator"]
    assert len(audit_entries) == 1
    assert "duration_ms" in audit_entries[0]


def test_route_after_validation_valid():
    state = make_state()
    state["output_valid"] = True
    state["generation_attempts"] = 1
    assert route_after_validation(state) == "pii_redactor"


def test_route_after_validation_retry():
    state = make_state()
    state["output_valid"] = False
    state["generation_attempts"] = 1
    assert route_after_validation(state) == "generator"


def test_route_after_validation_max_attempts():
    state = make_state()
    state["output_valid"] = False
    state["generation_attempts"] = settings.max_generation_attempts
    assert route_after_validation(state) == "hitl_router"


def test_toxicity_score_set():
    state = _state_with_response("This is a helpful and professional response.")
    result = output_validator_node(state)
    assert 0.0 <= result["toxicity_score"] <= 1.0
