from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import PipelineState
from app.graph.nodes.input_guard import input_guard_node
from app.graph.nodes.retriever import retriever_node
from app.graph.nodes.generator import generator_node
from app.graph.nodes.output_validator import output_validator_node, route_after_validation
from app.graph.nodes.pii_redactor import pii_redactor_node
from app.graph.nodes.grounding_checker import grounding_checker_node
from app.graph.nodes.hitl_router import hitl_router_node
from app.graph.nodes.audit_logger import audit_logger_node


def _route_after_input_guard(state: PipelineState) -> str:
    if state["input_guard_passed"]:
        return "retriever"
    return "audit_logger"


def build_graph(use_checkpointer: bool = True) -> object:
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("generator", generator_node)
    builder.add_node("output_validator", output_validator_node)
    builder.add_node("pii_redactor", pii_redactor_node)
    builder.add_node("grounding_checker", grounding_checker_node)
    builder.add_node("hitl_router", hitl_router_node)
    builder.add_node("audit_logger", audit_logger_node)

    # Entry
    builder.add_edge(START, "input_guard")

    # input_guard → retriever or audit_logger (short-circuit on block)
    builder.add_conditional_edges(
        "input_guard",
        _route_after_input_guard,
        {"retriever": "retriever", "audit_logger": "audit_logger"},
    )

    # Linear: retriever → generator → output_validator
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", "output_validator")

    # output_validator → generator (cycle) OR pii_redactor OR hitl_router
    builder.add_conditional_edges(
        "output_validator",
        route_after_validation,
        {
            "generator": "generator",
            "pii_redactor": "pii_redactor",
            "hitl_router": "hitl_router",
        },
    )

    # Linear: pii_redactor → grounding_checker → hitl_router
    builder.add_edge("pii_redactor", "grounding_checker")
    builder.add_edge("grounding_checker", "hitl_router")

    # hitl_router → audit_logger (always)
    builder.add_edge("hitl_router", "audit_logger")

    # audit_logger → END
    builder.add_edge("audit_logger", END)

    checkpointer = MemorySaver() if use_checkpointer else None
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
