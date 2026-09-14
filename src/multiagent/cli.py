"""Command-line entry point: run the graph on a task and print what happened."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .graph import build_graph
from .state import initial_state


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the planner/worker/reviewer LangGraph workflow.")
    parser.add_argument("task", nargs="?", default="Write a one-paragraph product update email.", help="Task for the agents to complete.")
    parser.add_argument("--max-revisions", type=int, default=1, help="How many times the worker may revise before escalation.")
    parser.add_argument("--live", action="store_true", help="Use a real Anthropic model (requires ANTHROPIC_API_KEY) instead of the offline mock.")
    parser.add_argument("--force-reject", choices=["true", "false"], default=None, help="Mock only: force every review to REJECT or APPROVE, to demonstrate a specific path deterministically.")
    args = parser.parse_args()

    mock = not args.live
    if not mock and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--live requires ANTHROPIC_API_KEY to be set (e.g. in a .env file).")

    force_reject = None
    if args.force_reject is not None:
        force_reject = args.force_reject == "true"

    graph = build_graph(mock=mock, force_reject=force_reject)
    state = initial_state(args.task, max_revisions=args.max_revisions)

    print(f"Task: {args.task}")
    print(f"Mode: {'LIVE (Anthropic API)' if not mock else 'MOCK (offline)'}")
    print(f"max_revisions: {args.max_revisions}")
    print("-" * 60)

    result = graph.invoke(state)

    print(f"\nFinal status: {result['status']}")
    print("\nReview history:")
    for record in result["review_history"]:
        print(f"  [{record['iteration']}] {record['decision'].upper()}: {record['feedback']}")

    if result["status"] == "approved":
        print("\nFinal output:\n")
        print(result["final_output"])
    else:
        print("\nEscalation reason:\n")
        print(result["escalation_reason"])


if __name__ == "__main__":
    main()
