from ulid import ULID
from app.graph.state import PipelineState, make_initial_state


def make_state(query: str = "What is the annual leave policy?", session_id: str = None) -> PipelineState:
    """Test helper — thin wrapper around the canonical make_initial_state factory.

    Using make_initial_state directly means any new PipelineState field added
    to state.py is automatically covered here with its correct default value,
    rather than causing confusing TypedDict validation errors in tests.
    """
    return make_initial_state(
        raw_query=query,
        session_id=session_id or str(ULID()),
    )
