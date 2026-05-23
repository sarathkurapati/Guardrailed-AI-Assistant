import json
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Response
from ulid import ULID

from app.config import settings
from app.graph.builder import get_graph
from app.graph.state import make_initial_state
from app.schemas.api import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    HITLResolveRequest,
    HITLQueueResponse,
    AuditResponse,
)
from app.services.presidio_engine import get_analyzer, get_anonymizer
from app.services.toxicity import get_toxicity_pipeline
from app.services.vector_store import get_vector_store
from app.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="Guardrailed Enterprise AI Assistant", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    logger.info("startup_begin")

    # 1. Load toxic-bert
    try:
        get_toxicity_pipeline()
        logger.info("startup_toxicity_ok")
    except Exception as e:
        logger.error("startup_toxicity_failed", error=str(e))

    # 2. Load Presidio engines
    try:
        get_analyzer()
        get_anonymizer()
        logger.info("startup_presidio_ok")
    except Exception as e:
        logger.error("startup_presidio_failed", error=str(e))

    # 3. Initialise ChromaDB
    try:
        get_vector_store()
        logger.info("startup_chroma_ok")
    except Exception as e:
        logger.error("startup_chroma_failed", error=str(e))

    # 4. Check Ollama reachability (non-blocking)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                logger.info("startup_ollama_ok")
            else:
                logger.warning("startup_ollama_unhealthy", status=resp.status_code)
    except Exception as e:
        logger.warning("startup_ollama_unreachable", error=str(e))

    # 5. Create data directories
    Path(settings.hitl_queue_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.audit_log_dir).mkdir(parents=True, exist_ok=True)
    logger.info("startup_directories_ready")

    # 6. Pre-build graph
    get_graph()
    logger.info("startup_graph_ready")

    logger.info("startup_complete")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, response: Response):
    session_id = request.session_id or str(ULID())
    initial_state = make_initial_state(
        raw_query=request.query,
        session_id=session_id,
    )

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("graph_invocation_error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Pipeline execution failed")

    status = final_state["route"]
    if not final_state["input_guard_passed"]:
        status = "blocked"
    elif final_state["route"] == "escalate_human":
        status = "pending_review"
        response.status_code = 202
    else:
        status = "ok"

    pii_redacted = len(final_state.get("pii_entities_found", [])) > 0

    return QueryResponse(
        session_id=session_id,
        response=final_state["final_response"],
        status=status,
        confidence_score=final_state.get("confidence_score"),
        grounding_passed=final_state.get("grounding_passed"),
        pii_redacted=pii_redacted,
        generation_attempts=final_state.get("generation_attempts"),
        block_reason=final_state.get("input_guard_reason") if not final_state["input_guard_passed"] else None,
    )


@app.get("/audit/{session_id}", response_model=AuditResponse)
async def get_audit(session_id: str):
    audit_file = Path(settings.audit_log_dir) / f"{session_id}.json"
    if not audit_file.exists():
        raise HTTPException(status_code=404, detail="Audit record not found")
    with open(audit_file) as f:
        data = json.load(f)
    return AuditResponse(session_id=session_id, audit_data=data)


@app.get("/health")
async def health_check():
    ollama_ok = False
    chroma_ok = False

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    try:
        store = get_vector_store()
        # Use get() with limit=1 — compatible with all langchain-chroma versions
        store.get(limit=1)
        chroma_ok = True
    except Exception:
        pass

    # Return plain dict to avoid Pydantic v2 serialisation quirks with booleans
    return {"status": "ok", "ollama_reachable": ollama_ok, "chroma_ready": chroma_ok}


@app.get("/hitl/queue", response_model=HITLQueueResponse)
async def get_hitl_queue():
    queue_dir = Path(settings.hitl_queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    sessions = [f.stem for f in queue_dir.glob("*.json") if not f.name.endswith("_resolved.json")]
    return HITLQueueResponse(pending_sessions=sessions, count=len(sessions))


@app.post("/hitl/resolve/{session_id}")
async def resolve_hitl(session_id: str, request: HITLResolveRequest):
    queue_dir = Path(settings.hitl_queue_dir)
    queue_file = queue_dir / f"{session_id}.json"

    if not queue_file.exists():
        raise HTTPException(status_code=404, detail="Session not found in HITL queue")

    with open(queue_file) as f:
        state = json.load(f)

    state["final_response"] = request.approved_response
    state["route"] = "auto_respond"
    state["resolved_by_human"] = True
    state["resolved_at"] = __import__("datetime").datetime.utcnow().isoformat()

    # Write resolved file and remove from queue
    resolved_file = queue_dir / f"{session_id}_resolved.json"
    with open(resolved_file, "w") as f:
        json.dump(state, f, indent=2)
    queue_file.unlink()

    # Also write to audit log
    audit_file = Path(settings.audit_log_dir) / f"{session_id}.json"
    if audit_file.exists():
        with open(audit_file) as f:
            audit = json.load(f)
        audit["human_approved_response"] = request.approved_response
        audit["resolved_at"] = state["resolved_at"]
        with open(audit_file, "w") as f:
            json.dump(audit, f, indent=2)

    logger.info("hitl_resolved", session_id=session_id)
    return {"status": "resolved", "session_id": session_id}
