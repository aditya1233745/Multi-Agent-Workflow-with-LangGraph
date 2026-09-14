"""
Run the workflow offline (no API key needed) and show the normal path:
worker drafts -> reviewer rejects once -> worker revises -> reviewer approves.

    python examples/run_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiagent.graph import build_graph
from multiagent.state import initial_state

if __name__ == "__main__":
    graph = build_graph(mock=True)
    state = initial_state("Summarise the benefits of unit testing for a new engineer.", max_revisions=1)

    result = graph.invoke(state)

    print(f"Final status: {result['status']}")
    for r in result["review_history"]:
        print(f"  [{r['iteration']}] {r['decision'].upper()}: {r['feedback']}")
    print()
    print(result["final_output"])
