import json
import time
from pathlib import Path
from datetime import datetime, timezone

from app.graph.state import PipelineState
from app.config import settings
from app.services.presidio_engine import analyze_text, anonymize_text
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_anonymize_query(query: str) -> str:
    """Anonymize PII in a query string before writing to the audit log.

    The input guardrail detects PII in the query but does NOT redact it
    (redaction only runs on LLM output). We must anonymize here so that
    raw PII values are never written to disk in the audit record.
    """
    try:
        results = analyze_text(query)
        if results:
            return anonymize_text(query, results)
    except Exception as e:
        logger.warning("audit_query_anonymization_error", error=str(e))
    return query


def audit_logger_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    # Append this node's own entry BEFORE writing the file so the
    # persisted record is complete and includes the audit_logger entry.
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "audit_logger",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": "logged",
    })

    audit_dir = Path(settings.audit_log_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_file = audit_dir / f"{state['session_id']}.json"

    # Anonymize the query before writing — never persist raw PII to disk.
    safe_query = _safe_anonymize_query(state["sanitized_query"])

    audit_record = {
        "session_id": state["session_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": safe_query,          # anonymized; raw_query is never written
        "final_response": state["final_response"],
        "route": state["route"],
        "input_guard_passed": state["input_guard_passed"],
        "injection_score": state["injection_score"],
        "pii_detected_in_input": state["pii_detected_in_input"],
        "generation_attempts": state["generation_attempts"],
        "output_valid": state["output_valid"],
        "toxicity_score": state["toxicity_score"],
        "bias_flag": state["bias_flag"],
        "grounding_passed": state["grounding_passed"],
        "ungrounded_claims_count": len(state["ungrounded_claims"]),
        "pii_entities_found": state["pii_entities_found"],
        "confidence_score": state["confidence_score"],
        "escalation_reason": state.get("escalation_reason"),
        "audit_trail": state["audit_trail"],
    }

    with open(audit_file, "w") as f:
        json.dump(audit_record, f, indent=2, default=str)

    logger.info(
        "audit_logged",
        session_id=state["session_id"],
        route=state["route"],
        file=str(audit_file),
    )

    return state
