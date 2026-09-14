"""
Wires the four nodes into a graph.

    planner -> worker -> reviewer --(approve)--> END
                  ^                 --(reject, budget left)--> worker
                  |                 --(reject, budget exhausted)--> escalation -> END
                  +---------------------------------+

The only conditional edge is out of `reviewer`. Every other edge is a
straight line. That single decision point -- "approve / revise / give up"
-- is deliberately the whole point of this project: it's what turns a
pipeline into a workflow that can correct itself, within a bound.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .llm import BaseAgentLLM, get_llm
from .nodes import escalation_node, make_planner_node, make_reviewer_node, make_worker_node
from .state import WorkflowState


def route_after_review(state: WorkflowState) -> str:
    """The one real decision in the graph. Reads state, decides where to go next."""
    if state["status"] == "approved":
        return "approved"
    if state["status"] == "escalated":
        return "escalate"
    if state["status"] == "revising":
        return "revise"
    # Should not happen; fail loud rather than silently looping forever.
    raise ValueError(f"Unexpected status after review: {state['status']!r}")


def build_graph(llm: BaseAgentLLM | None = None, mock: bool = True, force_reject: bool | None = None):
    """Build and compile the LangGraph StateGraph.

    Args:
        llm: pre-built LLM to reuse across all three agents. If None, one is
            created from `mock`/`force_reject`.
        mock: use the deterministic offline mock instead of a real model.
        force_reject: only used by the mock -- force every review to
            APPROVE (False) or REJECT (True), useful for tests and for
            demonstrating the escalation path deterministically.
    """
    shared_llm = llm or get_llm(mock=mock, force_reject=force_reject)

    builder = StateGraph(WorkflowState)

    builder.add_node("planner", make_planner_node(shared_llm))
    builder.add_node("worker", make_worker_node(shared_llm))
    builder.add_node("reviewer", make_reviewer_node(shared_llm))
    builder.add_node("escalation", escalation_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", "reviewer")

    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "approved": END,
            "revise": "worker",
            "escalate": "escalation",
        },
    )
    builder.add_edge("escalation", END)

    return builder.compile()
