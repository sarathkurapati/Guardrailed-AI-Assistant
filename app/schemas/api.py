from typing import Optional, List, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    response: str
    status: str  # "ok" | "blocked" | "pending_review"
    confidence_score: Optional[float] = None
    grounding_passed: Optional[bool] = None
    pii_redacted: Optional[bool] = None
    generation_attempts: Optional[int] = None
    block_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chroma_ready: bool


class HITLResolveRequest(BaseModel):
    approved_response: str


class HITLQueueResponse(BaseModel):
    pending_sessions: List[str]
    count: int


class AuditResponse(BaseModel):
    session_id: str
    audit_data: Any
