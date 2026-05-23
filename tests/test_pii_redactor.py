import pytest
from tests.conftest import make_state
from app.graph.nodes.pii_redactor import pii_redactor_node


def _state_with_raw(raw: str):
    state = make_state()
    state["raw_response"] = raw
    return state


def test_email_redacted():
    state = _state_with_raw("Contact john.smith@acmecorp.com for details.")
    result = pii_redactor_node(state)
    assert "john.smith@acmecorp.com" not in result["redacted_response"]
    assert "<EMAIL_ADDRESS>" in result["redacted_response"]


def test_person_name_redacted():
    state = _state_with_raw("Please contact John Smith for assistance.")
    result = pii_redactor_node(state)
    assert len(result["pii_entities_found"]) > 0


def test_no_pii_passthrough():
    safe_text = "Annual leave must be submitted 5 working days in advance."
    state = _state_with_raw(safe_text)
    result = pii_redactor_node(state)
    assert result["redacted_response"] == safe_text
    assert result["pii_entities_found"] == []


def test_entities_logged_not_values():
    state = _state_with_raw("Contact sarah@example.com or call +44 7911 123456.")
    result = pii_redactor_node(state)
    for entity in result["pii_entities_found"]:
        assert "entity_type" in entity
        assert "start" in entity
        assert "end" in entity
        # The actual PII value must NOT be stored
        assert "value" not in entity


def test_redacted_response_not_empty_on_no_pii():
    state = _state_with_raw("The policy requires 5 days notice.")
    result = pii_redactor_node(state)
    assert result["redacted_response"] != ""


def test_audit_trail_appended():
    state = _state_with_raw("Contact test@example.com for help.")
    result = pii_redactor_node(state)
    entries = [e for e in result["audit_trail"] if e["node"] == "pii_redactor"]
    assert len(entries) == 1
    assert "duration_ms" in entries[0]


def test_phone_number_redacted():
    # Presidio's PhoneRecognizer requires context keywords (e.g. "phone", "call",
    # "contact") to boost confidence above 0.5 for most formats.
    # "(555) 123-4567" with "phone" context reliably scores 0.75.
    state = _state_with_raw("My phone number is (555) 123-4567, please call anytime.")
    result = pii_redactor_node(state)
    assert "<PHONE_NUMBER>" in result["redacted_response"]
