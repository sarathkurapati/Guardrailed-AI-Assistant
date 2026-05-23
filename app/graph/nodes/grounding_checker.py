import re
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.graph.state import PipelineState
from app.services.llm import get_llm
from app.services.embeddings import get_embeddings
from app.config import settings
from app.utils.logging import get_logger
from app.utils.similarity import cosine_similarity

logger = get_logger(__name__)

CLAIM_EXTRACTION_PROMPT = """Extract all distinct factual claims from the following text.
Return ONLY a JSON array of strings, no other text.
Text: {response}"""


def _extract_claims(response: str) -> list[str]:
    prompt = CLAIM_EXTRACTION_PROMPT.format(response=response)
    try:
        def _call():
            return get_llm().invoke(prompt)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            raw = future.result(timeout=settings.llm_timeout)

        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c) for c in claims if c]
        return []
    except (FuturesTimeoutError, TimeoutError):
        logger.warning("claim_extraction_timeout")
    except json.JSONDecodeError:
        logger.warning("claim_extraction_json_error", raw_snippet=raw[:100] if "raw" in dir() else "")

    # Fallback: split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _embed_docs(docs) -> list[list[float]]:
    emb = get_embeddings()
    texts = [d.page_content for d in docs]
    return emb.embed_documents(texts)


def grounding_checker_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    # Use raw_response (pre-redaction) so that PII placeholder tokens
    # like <EMAIL_ADDRESS> don't degrade cosine similarity against docs.
    response = state["raw_response"]
    docs = state["retrieved_docs"]

    # If generator correctly refused, treat as grounded
    if "i don't have enough information" in response.lower():
        state["grounding_passed"] = True
        state["ungrounded_claims"] = []
        _append_audit(state, start, True, "generator_refused", [], 0, 0)
        return state

    # No docs → all claims ungrounded
    if not docs:
        state["grounding_passed"] = False
        state["ungrounded_claims"] = []
        _append_audit(state, start, False, "no_retrieved_docs", [], 0, 0)
        return state

    claims = _extract_claims(response)
    if not claims:
        state["grounding_passed"] = True
        state["ungrounded_claims"] = []
        _append_audit(state, start, True, "no_claims_extracted", [], 0, 0)
        return state

    # Cache doc embeddings for this pipeline run
    try:
        doc_embeddings = _embed_docs(docs)
        emb = get_embeddings()
        ungrounded = []

        for claim in claims:
            claim_vec = emb.embed_query(claim)
            max_sim = max(cosine_similarity(claim_vec, dv) for dv in doc_embeddings)
            if max_sim < settings.grounding_similarity_threshold:
                ungrounded.append(claim)

        ungrounded_ratio = len(ungrounded) / len(claims)
        passed = ungrounded_ratio <= settings.grounding_failure_ratio

        state["grounding_passed"] = passed
        state["ungrounded_claims"] = ungrounded

        logger.info(
            "grounding_check_result",
            session_id=state["session_id"],
            total_claims=len(claims),
            ungrounded=len(ungrounded),
            ratio=round(ungrounded_ratio, 3),
            passed=passed,
        )
        _append_audit(state, start, passed, f"{len(ungrounded)}/{len(claims)} ungrounded", ungrounded, len(claims), len(ungrounded))

    except Exception as e:
        logger.error("grounding_checker_error", session_id=state["session_id"], error=str(e))
        state["grounding_passed"] = True  # fail open
        state["ungrounded_claims"] = []
        _append_audit(state, start, True, f"error_fail_open: {e}", [], 0, 0)

    return state


def _append_audit(state, start, passed, reason, ungrounded, total, n_ungrounded):
    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "grounding_checker",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": "passed" if passed else "failed",
        "reason": reason,
        "ungrounded_claims": ungrounded,
        "total_claims": total,
        "ungrounded_count": n_ungrounded,
    })
