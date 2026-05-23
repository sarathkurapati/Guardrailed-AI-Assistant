import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from tests.conftest import make_state
from app.graph.nodes.grounding_checker import grounding_checker_node
from app.utils.similarity import cosine_similarity


def _state_with_docs_and_response(response: str, docs: list):
    state = make_state()
    # grounding_checker reads raw_response (pre-redaction) to avoid
    # PII placeholder tokens degrading cosine similarity against docs.
    state["raw_response"] = response
    state["retrieved_docs"] = docs
    return state


def test_refusal_treated_as_grounded():
    state = _state_with_docs_and_response(
        "I don't have enough information to answer that.",
        [Document(page_content="Some doc content")]
    )
    result = grounding_checker_node(state)
    assert result["grounding_passed"] is True
    assert result["ungrounded_claims"] == []


def test_no_docs_fails_grounding():
    state = _state_with_docs_and_response(
        "The company was founded in 1995 and has 500 employees.",
        []
    )
    result = grounding_checker_node(state)
    assert result["grounding_passed"] is False


def test_audit_trail_appended():
    state = _state_with_docs_and_response(
        "I don't have enough information to answer that.",
        []
    )
    result = grounding_checker_node(state)
    entries = [e for e in result["audit_trail"] if e["node"] == "grounding_checker"]
    assert len(entries) == 1
    assert "duration_ms" in entries[0]


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert cosine_similarity(a, b) == 0.0


def test_grounding_state_fields_set():
    state = _state_with_docs_and_response(
        "I don't have enough information to answer that.",
        []
    )
    result = grounding_checker_node(state)
    assert "grounding_passed" in result
    assert "ungrounded_claims" in result
    assert isinstance(result["ungrounded_claims"], list)
