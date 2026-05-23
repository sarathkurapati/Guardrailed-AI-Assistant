import time
from datetime import datetime, timezone

from app.graph.state import PipelineState
from app.services.vector_store import get_vector_store
from app.utils.logging import get_logger

logger = get_logger(__name__)


def retriever_node(state: PipelineState) -> PipelineState:
    state = dict(state)
    start = time.monotonic()

    query = state["sanitized_query"]
    try:
        store = get_vector_store()
        docs = store.similarity_search(query, k=5)
        state["retrieved_docs"] = docs
        logger.info(
            "retriever_results",
            session_id=state["session_id"],
            count=len(docs),
            sources=[d.metadata.get("source", "unknown") for d in docs],
        )
    except Exception as e:
        logger.error("retriever_error", session_id=state["session_id"], error=str(e))
        state["retrieved_docs"] = []

    duration_ms = round((time.monotonic() - start) * 1000, 2)
    state["audit_trail"].append({
        "node": "retriever",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "decision": "retrieved",
        "reason": f"found {len(state['retrieved_docs'])} documents",
        "doc_count": len(state["retrieved_docs"]),
    })

    return state
