# Failure mode: what happens when the reviewer rejects twice

## The naive version, and why it breaks

The first design I tried was simpler than what's in this repo: on rejection,
just route straight back to the worker, unconditionally, forever, until the
reviewer approves. That's the obvious reading of "the reviewer can send
work back."

It breaks in two ways, and both are worth naming because they show up
differently:

**1. It can genuinely never terminate.** If the reviewer's bar is
unreachable — e.g. the task itself is underspecified, or the plan from step
one is wrong and every draft built on it inherits the flaw — the worker
keeps producing drafts that are dutifully revised versions of a
fundamentally wrong thing, and the reviewer keeps rejecting them. Nothing
in an unconditional `reject → worker` edge ever stops this. In a real
deployment this is a hung process burning API calls until someone notices
the bill or the timeout.

**2. Even if it does terminate, silent success/failure is worse than it
looks.** Say you cap it with a plain retry loop outside the graph ("try 5
times, then just ship the last draft anyway"). Now the graph *always*
returns a `final_output`, whether or not the reviewer ever approved it.
Nothing downstream can tell the difference between "the reviewer signed off
on this" and "we gave up and shipped whatever we had." That's a worse
failure than an explicit stop, because it fails silently — a rejected,
known-flawed draft goes out the same door as an approved one.

## What this repo does instead

Two changes, both visible in `nodes.py::make_reviewer_node` and
`graph.py::route_after_review`:

1. **A revision budget lives in state** (`max_revisions`, default `1`,
   checked against `revision_count`). The reviewer node — not some
   external retry wrapper — increments the counter and compares it to the
   budget on every rejection. This keeps "should we keep going" as part of
   the workflow's own state, inspectable and testable, rather than a
   side-channel loop counter bolted on around the graph.

2. **The two outcomes are distinguishable in the returned state.** A
   rejection within budget sets `status="revising"` and the conditional
   edge sends control back to `worker`. A rejection that exceeds the
   budget sets `status="escalated"` and routes to a dedicated
   `escalation` node instead of back to the worker. The escalation node
   does not attempt another draft. It writes `final_output=None` and
   fills `escalation_reason` with the full `review_history` (every
   iteration's decision and feedback) plus the last, still-rejected draft.
   Anything consuming this graph's output can check `status` and knows
   immediately: `"approved"` means ship it, `"escalated"` means a human
   needs to look at `escalation_reason` before anything goes out.

With the default `max_revisions=1`, the sequence for the "rejects twice"
case is:

```
planner → worker → reviewer (REJECT #1, revision_count=1, budget=1, 1<=1 -> revise)
       → worker → reviewer (REJECT #2, revision_count=2, budget=1, 2>1  -> escalate)
       → escalation → END (status="escalated", final_output=None)
```

Two review attempts total, not an unbounded loop, and the graph ends in a
state that's explicitly marked as unresolved rather than quietly shipping
the second rejected draft. `examples/run_escalation_example.py` and
`tests/test_graph.py::test_reject_twice_escalates_and_does_not_loop_forever`
both exercise this path directly, forcing the mock reviewer to reject
every time so the escalation branch is deterministic rather than something
you have to get lucky to trigger.

## What I'd still want before trusting this in production

- **Escalation should be an action, not just a state.** Right now
  `escalation_reason` is just a string sitting in the returned state. In a
  real system this node would page a human, open a ticket, or write to a
  queue — the graph shape supports that (it's one more node before `END`),
  but this repo stops at "make the failure visible and inspectable,"
  which is as far as the assignment's scope goes.
- **A fixed numeric budget is a blunt instrument.** `max_revisions=1`
  treats "rejected because of a typo" the same as "rejected because the
  whole approach is wrong." A more nuanced version would let the reviewer
  emit a severity alongside its feedback and let that influence whether
  another revision is worth attempting versus escalating immediately —
  but that's a reviewer-prompt-design problem, not a graph-shape problem,
  so I kept it out of scope here.
