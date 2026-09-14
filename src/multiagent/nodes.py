"""
The three specialised agents plus the escalation node they fall back to.

Each node function has the LangGraph node signature: it takes the full
WorkflowState and returns a dict of the fields it wants to update. LangGraph
merges that dict into the running state before calling the next node.
"""

from __future__ import annotations

from .llm import BaseAgentLLM
from .state import ReviewRecord, WorkflowState

PLANNER_SYSTEM = (
    "You are the PLANNER agent in a multi-agent workflow. Given a task, "
    "produce a short, numbered plan (3-5 steps) that a WORKER agent will "
    "follow to produce a draft. Do not write the draft yourself."
)

WORKER_SYSTEM = (
    "You are the WORKER agent in a multi-agent workflow. Given a task and a "
    "plan, produce a draft response that follows the plan. If revision "
    "feedback from a REVIEWER agent is included, you MUST address every "
    "point in the feedback in your revised draft."
)

REVIEWER_SYSTEM = (
    "You are the REVIEWER agent in a multi-agent workflow. Given a task and "
    "a draft, decide whether the draft is acceptable. Respond with the "
    "first line being exactly APPROVE or REJECT, followed by a line "
    "'Feedback: ...' explaining your decision. Be specific and actionable "
    "in feedback so a WORKER agent can act on it."
)


def make_planner_node(llm: BaseAgentLLM):
    def planner_node(state: WorkflowState) -> dict:
        user = f"Task: {state['task']}\n\nProduce the plan."
        response = llm.complete(PLANNER_SYSTEM, user)
        return {
            "plan": response.text.strip(),
            "status": "drafting",
        }

    return planner_node


def make_worker_node(llm: BaseAgentLLM):
    def worker_node(state: WorkflowState) -> dict:
        parts = [f"Task: {state['task']}", f"Plan:\n{state['plan']}"]

        if state["review_history"]:
            last = state["review_history"][-1]
            if last["decision"] == "reject":
                parts.append(
                    "REVISION FEEDBACK from reviewer (address all of this):\n"
                    f"{last['feedback']}"
                )
                if state["draft"]:
                    parts.append(f"Previous draft:\n{state['draft']}")

        user = "\n\n".join(parts)
        response = llm.complete(WORKER_SYSTEM, user)
        return {
            "draft": response.text.strip(),
            "status": "reviewing",
        }

    return worker_node


def make_reviewer_node(llm: BaseAgentLLM):
    def reviewer_node(state: WorkflowState) -> dict:
        user = (
            f"Task: {state['task']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Draft to review:\n{state['draft']}"
        )
        if state["review_history"]:
            user += "\n\n(Note: this draft is a revision after prior feedback.)"

        response = llm.complete(REVIEWER_SYSTEM, user)
        decision, feedback = _parse_review(response.text)

        iteration = len(state["review_history"]) + 1
        record: ReviewRecord = {
            "iteration": iteration,
            "decision": decision,
            "feedback": feedback,
        }
        history = state["review_history"] + [record]

        if decision == "approve":
            return {
                "review_history": history,
                "status": "approved",
                "final_output": state["draft"],
            }

        # rejected
        new_revision_count = state["revision_count"] + 1
        if new_revision_count > state["max_revisions"]:
            # This is the failure mode: the reviewer has now rejected more
            # times than we're willing to retry. Stop looping and hand off
            # to a human instead of burning more turns.
            return {
                "review_history": history,
                "revision_count": new_revision_count,
                "status": "escalated",
                "escalation_reason": (
                    f"Reviewer rejected {new_revision_count} time(s), exceeding "
                    f"max_revisions={state['max_revisions']}. Last feedback: {feedback}"
                ),
            }

        return {
            "review_history": history,
            "revision_count": new_revision_count,
            "status": "revising",
        }

    return reviewer_node


def escalation_node(state: WorkflowState) -> dict:
    """
    Terminal node reached only when the reviewer has rejected more times
    than max_revisions allows. It doesn't try to fix anything -- it packages
    up what's known and stops, so a human can pick it up with full context
    instead of the graph silently looping or silently shipping a rejected
    draft.
    """
    summary_lines = [
        f"Escalated after {state['revision_count']} rejection(s) "
        f"(max_revisions={state['max_revisions']}).",
        "",
        "Review history:",
    ]
    for record in state["review_history"]:
        summary_lines.append(
            f"  [{record['iteration']}] {record['decision'].upper()}: {record['feedback']}"
        )
    summary_lines.append("")
    summary_lines.append(f"Last draft (unapproved):\n{state['draft']}")

    return {
        "final_output": None,
        "escalation_reason": "\n".join(summary_lines),
    }


def _parse_review(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    decision = "reject"
    feedback = text.strip()

    if lines:
        first = lines[0].upper()
        if first.startswith("APPROVE"):
            decision = "approve"
        elif first.startswith("REJECT"):
            decision = "reject"

        for line in lines[1:]:
            if line.lower().startswith("feedback:"):
                feedback = line.split(":", 1)[1].strip()
                break

    return decision, feedback
