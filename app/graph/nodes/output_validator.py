import re
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.graph.state import PipelineState
from app.services.toxicity import score_toxicity
from app.services.llm import get_llm
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _bias_check(response: str) -> bool:
    prompt = (
        "Does the following response contain stereotypes, discriminatory language, "
        "or unfair bias? Respond only: YES or NO.\n\n"
        f"{response}"
    )
    try:
        def _call():
            return get_llm().invoke(prompt)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            result = future.result(timeout=settings.llm_timeout)

        return result.strip().upper().startswith("YES")
    except (FuturesTimeoutError, TimeoutError):
        logger.warning("bias_check_timeout")
        return False  # fail open
    except Exception as e:
        logger.warning("bias_check_error", error=str(e))
        return False


def output_validator_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    response = state["raw_response"]
    failure_reason: str | None = None
    toxicity_score = 0.0
    bias_flag = False

    # 1. Empty/short check first (before calling toxic-bert on empty string)
    if not response or len(response.strip()) < 10:
        failure_reason = "empty_or_too_short"
        state["output_valid"] = False
        state["toxicity_score"] = 0.0
        state["bias_flag"] = False
        state["validation_failure_reason"] = failure_reason
        _append_audit(state, start, False, failure_reason, 0.0, False)
        return state

    # 2. Length sanity check
    if len(response) > 2000:
        failure_reason = "response_too_long"
        state["output_valid"] = False
        state["toxicity_score"] = 0.0
        state["bias_flag"] = False
        state["validation_failure_reason"] = failure_reason
        _append_audit(state, start, False, failure_reason, 0.0, False)
        return state

    # 3. Toxicity check
    toxicity_score = score_toxicity(response)
    if toxicity_score > settings.toxicity_hard_block_threshold:
        failure_reason = f"toxicity_hard_block: score={toxicity_score:.3f}"
    elif toxicity_score > settings.toxicity_threshold:
        failure_reason = f"toxicity_threshold_exceeded: score={toxicity_score:.3f}"

    # 4. Bias check (only if no hard toxicity block)
    if not failure_reason:
        bias_flag = _bias_check(response)
        if bias_flag:
            failure_reason = "bias_detected"

    state["toxicity_score"] = toxicity_score
    state["bias_flag"] = bias_flag
    state["output_valid"] = failure_reason is None
    state["validation_failure_reason"] = failure_reason

    _append_audit(state, start, failure_reason is None, failure_reason, toxicity_score, bias_flag)
    return state


def _append_audit(state, start, valid, reason, toxicity, bias):
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "output_validator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": "valid" if valid else "invalid",
        "reason": reason,
        "toxicity_score": toxicity,
        "bias_flag": bias,
        "generation_attempts": state["generation_attempts"],
    })


def route_after_validation(state: PipelineState) -> str:
    if state["output_valid"]:
        return "pii_redactor"
    elif state["generation_attempts"] >= settings.max_generation_attempts:
        return "hitl_router"
    else:
        return "generator"
