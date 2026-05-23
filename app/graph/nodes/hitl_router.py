import json
import time
from pathlib import Path
from datetime import datetime, timezone

from app.graph.state import PipelineState
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _compute_confidence(state: PipelineState) -> float:
    confidence = 1.0
    if not state["grounding_passed"]:
        confidence -= 0.4
    if state["toxicity_score"] > 0.3:
        confidence -= 0.3
    if state["pii_detected_in_input"]:
        confidence -= 0.1
    if state["generation_attempts"] > 1:
        confidence -= 0.1 * state["generation_attempts"]
    return max(0.0, confidence)


def hitl_router_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    confidence = _compute_confidence(state)
    state["confidence_score"] = confidence

    # Escalate if already flagged (from output_validator 3-attempt exhaustion)
    # or if confidence is below threshold
    should_escalate = (
        state.get("route") == "escalate_human"
        or confidence < settings.hitl_confidence_threshold
    )

    if should_escalate:
        state["route"] = "escalate_human"
        if not state.get("escalation_reason"):
            state["escalation_reason"] = f"low_confidence: {confidence:.3f}"
        state["final_response"] = (
            f"Your query has been escalated for human review. "
            f"Reference ID: {state['session_id']}"
        )
        _persist_to_queue(state)
        logger.info(
            "hitl_escalated",
            session_id=state["session_id"],
            confidence=confidence,
            reason=state["escalation_reason"],
        )
    else:
        state["route"] = "auto_respond"
        state["final_response"] = state["redacted_response"]
        logger.info(
            "hitl_auto_respond",
            session_id=state["session_id"],
            confidence=confidence,
        )

    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "hitl_router",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": state["route"],
        "reason": state.get("escalation_reason") or "auto_respond",
        "confidence_score": confidence,
    })

    return state


def _persist_to_queue(state: PipelineState) -> None:
    queue_dir = Path(settings.hitl_queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_file = queue_dir / f"{state['session_id']}.json"

    serializable_state = {
        k: v for k, v in state.items()
        if k != "retrieved_docs"  # Document objects not JSON-serializable
    }
    serializable_state["retrieved_docs_count"] = len(state.get("retrieved_docs", []))

    with open(queue_file, "w") as f:
        json.dump(serializable_state, f, indent=2, default=str)
