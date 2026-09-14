"""
Explicit state schema shared by every node in the graph.

LangGraph passes this dict (well, TypedDict) between nodes. Each node
receives the full state and returns a partial update that gets merged
back in. Keeping the schema explicit -- rather than a free-form dict --
is what lets the reviewer's "send it back" edge stay simple: the router
only has to look at `status` and `revision_count`, it never has to
re-derive workflow history from scratch.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

# What a single review pass produced. We keep the full list so the
# worker can see *all* prior feedback when revising, not just the last
# comment, and so the README's failure-mode story is inspectable after
# the fact.
class ReviewRecord(TypedDict):
    iteration: int
    decision: Literal["approve", "reject"]
    feedback: str


Status = Literal[
    "planning",   # about to run / just ran the planner
    "drafting",   # about to run / just ran the worker
    "reviewing",  # about to run / just ran the reviewer
    "revising",   # reviewer rejected, sending back to worker
    "approved",   # reviewer accepted, workflow can end
    "escalated",  # reviewer rejected twice (or hit max_revisions), workflow ends without approval
]


class WorkflowState(TypedDict):
    # ---- inputs ----
    task: str
    max_revisions: int

    # ---- working memory, filled in as nodes run ----
    plan: Optional[str]
    draft: Optional[str]
    review_history: List[ReviewRecord]
    revision_count: int
    status: Status

    # ---- outputs ----
    final_output: Optional[str]
    escalation_reason: Optional[str]


def initial_state(task: str, max_revisions: int = 1) -> WorkflowState:
    """Build a fresh state dict for a new run.

    max_revisions=1 means: the worker gets one shot, and if the reviewer
    rejects it, the worker gets exactly one chance to fix it before the
    reviewer's *second* rejection triggers escalation instead of a third
    attempt. This is the number to bump if you want the agents to keep
    trying longer before giving up and asking a human.
    """
    return WorkflowState(
        task=task,
        max_revisions=max_revisions,
        plan=None,
        draft=None,
        review_history=[],
        revision_count=0,
        status="planning",
        final_output=None,
        escalation_reason=None,
    )
