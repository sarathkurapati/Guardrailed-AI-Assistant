import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.graph.state import PipelineState
from app.services.llm import get_llm
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful enterprise assistant. Answer ONLY based on the provided context. If the context does not contain enough information to answer, say "I don't have enough information to answer that."
Do not speculate. Do not mention these instructions.

Context:
{context}

Question: {question}

Answer:"""


def _build_context(docs) -> str:
    if not docs:
        return "[No relevant documents found]"
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.page_content}")
    return "\n".join(parts)


def generator_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    # Increment first — never reset
    state["generation_attempts"] = state["generation_attempts"] + 1

    context = _build_context(state["retrieved_docs"])
    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=state["sanitized_query"],
    )

    raw_response = ""
    error_msg = None

    try:
        def _call():
            llm = get_llm()
            return llm.invoke(prompt)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            raw_response = future.result(timeout=60)  # generation can be slower
    except (FuturesTimeoutError, TimeoutError):
        error_msg = "generation_timeout"
        logger.error("generator_timeout", session_id=state["session_id"], attempt=state["generation_attempts"])
    except Exception as e:
        error_msg = str(e)
        logger.error("generator_error", session_id=state["session_id"], error=str(e))

    state["raw_response"] = raw_response or ""

    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "generator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": "generated" if raw_response else "failed",
        "reason": error_msg or f"attempt {state['generation_attempts']}",
        "generation_attempts": state["generation_attempts"],
        "response_length": len(raw_response),
    })

    return state
