import pytest
from tests.conftest import make_state
from app.graph.nodes.hitl_router import hitl_router_node, _compute_confidence


def _state_high_confidence():
    state = make_state()
    state["grounding_passed"] = True
    state["toxicity_score"] = 0.1
    state["pii_detected_in_input"] = False
    state["generation_attempts"] = 1
    state["redacted_response"] = "The annual leave entitlement is 20 days per year."
    state["route"] = "auto_respond"
    return state


def _state_low_confidence():
    state = make_state()
    state["grounding_passed"] = False
    state["toxicity_score"] = 0.4
    state["pii_detected_in_input"] = True
    state["generation_attempts"] = 3
    state["redacted_response"] = "Some response."
    state["route"] = "auto_respond"
    return state


def test_high_confidence_auto_responds():
    state = _state_high_confidence()
    result = hitl_router_node(state)
    assert result["route"] == "auto_respond"
    assert result["final_response"] == "The annual leave entitlement is 20 days per year."


def test_low_confidence_escalates():
    state = _state_low_confidence()
    result = hitl_router_node(state)
    assert result["route"] == "escalate_human"
    assert "Reference ID" in result["final_response"]


def test_pre_flagged_escalation_preserved():
    state = _state_high_confidence()
    state["route"] = "escalate_human"
    state["escalation_reason"] = "output_failed_after_3_attempts"
    result = hitl_router_node(state)
    assert result["route"] == "escalate_human"


def test_confidence_score_set():
    state = _state_high_confidence()
    result = hitl_router_node(state)
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_compute_confidence_formula():
    state = make_state()
    state["grounding_passed"] = True
    state["toxicity_score"] = 0.0
    state["pii_detected_in_input"] = False
    state["generation_attempts"] = 1
    assert _compute_confidence(state) == pytest.approx(1.0, abs=1e-6)


def test_compute_confidence_grounding_penalty():
    state = make_state()
    state["grounding_passed"] = False
    state["toxicity_score"] = 0.0
    state["pii_detected_in_input"] = False
    state["generation_attempts"] = 1
    assert _compute_confidence(state) == pytest.approx(0.6, abs=1e-6)


def test_audit_trail_appended():
    state = _state_high_confidence()
    result = hitl_router_node(state)
    entries = [e for e in result["audit_trail"] if e["node"] == "hitl_router"]
    assert len(entries) == 1
    assert "duration_ms" in entries[0]
    assert "confidence_score" in entries[0]


def test_escalation_writes_session_id_to_response():
    state = _state_low_confidence()
    state["session_id"] = "test-session-123"
    result = hitl_router_node(state)
    assert "test-session-123" in result["final_response"]
