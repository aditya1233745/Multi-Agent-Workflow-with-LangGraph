"""
Force the reviewer to reject every draft, to deterministically demonstrate
the escalation path (reviewer rejects twice -> graph stops and hands off
to a human instead of looping forever).

    python examples/run_escalation_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiagent.graph import build_graph
from multiagent.state import initial_state

if __name__ == "__main__":
    # force_reject=True makes the mock reviewer reject unconditionally, so
    # with max_revisions=1 the graph runs: worker -> reject -> worker ->
    # reject again -> escalate. No infinite loop, no silent failure.
    graph = build_graph(mock=True, force_reject=True)
    state = initial_state("Draft a legal disclaimer for a new app.", max_revisions=1)

    result = graph.invoke(state)

    print(f"Final status: {result['status']}")
    for r in result["review_history"]:
        print(f"  [{r['iteration']}] {r['decision'].upper()}: {r['feedback']}")
    print()
    print(result["escalation_reason"])
