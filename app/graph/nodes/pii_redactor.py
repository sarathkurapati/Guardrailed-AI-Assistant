import time
from datetime import datetime, timezone

from app.graph.state import PipelineState
from app.services.presidio_engine import analyze_text, anonymize_text
from app.utils.logging import get_logger

logger = get_logger(__name__)


def pii_redactor_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    response = state["raw_response"]
    redacted = response
    entities_found = []

    try:
        results = analyze_text(response)
        if results:
            redacted = anonymize_text(response, results)
            entities_found = [
                {
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": round(r.score, 3),
                }
                for r in results
            ]
            # Log types and positions only — never the actual PII values
            logger.info(
                "pii_redacted",
                session_id=state["session_id"],
                count=len(results),
                types=[r.entity_type for r in results],
                positions=[(r.start, r.end) for r in results],
            )
    except Exception as e:
        logger.error("pii_redactor_error", session_id=state["session_id"], error=str(e))
        redacted = response  # pass-through on error

    state["redacted_response"] = redacted
    state["pii_entities_found"] = entities_found

    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "pii_redactor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": "redacted" if entities_found else "pass_through",
        "reason": f"{len(entities_found)} entities found",
        "entity_types": [e["entity_type"] for e in entities_found],
    })

    return state
