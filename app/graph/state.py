from typing import TypedDict, Optional, Literal, List
from langchain_core.documents import Document


class PipelineState(TypedDict):
    # Input
    session_id: str
    raw_query: str

    # After input guard
    sanitized_query: str
    input_guard_passed: bool
    input_guard_reason: Optional[str]
    injection_score: float
    pii_detected_in_input: bool

    # After retriever
    retrieved_docs: List[Document]

    # After generator
    raw_response: str
    generation_attempts: int

    # After output validator
    output_valid: bool
    toxicity_score: float
    bias_flag: bool
    validation_failure_reason: Optional[str]

    # After PII redactor
    redacted_response: str
    pii_entities_found: List[dict]

    # After grounding checker
    grounding_passed: bool
    ungrounded_claims: List[str]

    # After HITL router
    route: Literal["auto_respond", "escalate_human"]
    confidence_score: float
    escalation_reason: Optional[str]

    # Final
    final_response: str
    audit_trail: List[dict]


def make_initial_state(raw_query: str, session_id: str) -> PipelineState:
    return PipelineState(
        session_id=session_id,
        raw_query=raw_query,
        sanitized_query="",
        input_guard_passed=False,
        input_guard_reason=None,
        injection_score=0.0,
        pii_detected_in_input=False,
        retrieved_docs=[],
        raw_response="",
        generation_attempts=0,
        output_valid=False,
        toxicity_score=0.0,
        bias_flag=False,
        validation_failure_reason=None,
        redacted_response="",
        pii_entities_found=[],
        grounding_passed=False,
        ungrounded_claims=[],
        route="auto_respond",
        confidence_score=1.0,
        escalation_reason=None,
        final_response="",
        audit_trail=[],
    )
