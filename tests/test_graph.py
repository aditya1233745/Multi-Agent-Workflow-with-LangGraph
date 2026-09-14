import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiagent.graph import build_graph
from multiagent.state import initial_state


def test_approve_on_first_pass_ends_approved():
    graph = build_graph(mock=True, force_reject=False)
    state = initial_state("Do the thing.", max_revisions=1)
    result = graph.invoke(state)

    assert result["status"] == "approved"
    assert result["final_output"] is not None
    assert len(result["review_history"]) == 1
    assert result["review_history"][0]["decision"] == "approve"


def test_reject_then_approve_takes_the_revise_edge():
    # default mock behavior: reject first pass, approve the revision
    graph = build_graph(mock=True, force_reject=None)
    state = initial_state("Do the thing.", max_revisions=1)
    result = graph.invoke(state)

    assert result["status"] == "approved"
    assert len(result["review_history"]) == 2
    assert result["review_history"][0]["decision"] == "reject"
    assert result["review_history"][1]["decision"] == "approve"
    assert result["revision_count"] == 1


def test_reject_twice_escalates_and_does_not_loop_forever():
    graph = build_graph(mock=True, force_reject=True)
    state = initial_state("Do the thing.", max_revisions=1)
    result = graph.invoke(state)

    assert result["status"] == "escalated"
    assert result["final_output"] is None
    assert result["escalation_reason"] is not None
    # exactly max_revisions + 1 review attempts: initial + 1 retry, then stop
    assert len(result["review_history"]) == 2
    assert all(r["decision"] == "reject" for r in result["review_history"])


def test_max_revisions_zero_escalates_on_first_rejection():
    graph = build_graph(mock=True, force_reject=True)
    state = initial_state("Do the thing.", max_revisions=0)
    result = graph.invoke(state)

    assert result["status"] == "escalated"
    assert len(result["review_history"]) == 1


def test_higher_max_revisions_allows_more_retries_before_escalating():
    graph = build_graph(mock=True, force_reject=True)
    state = initial_state("Do the thing.", max_revisions=3)
    result = graph.invoke(state)

    assert result["status"] == "escalated"
    # initial attempt + 3 retries = 4 rejections before giving up
    assert len(result["review_history"]) == 4
