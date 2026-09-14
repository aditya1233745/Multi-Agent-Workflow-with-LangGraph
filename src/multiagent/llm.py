"""
Thin wrapper around the model calls so the graph/nodes don't care whether
they're talking to a real Anthropic model or a deterministic mock.

Why a mock exists at all: this repo needs to be runnable and demonstrably
correct (including the "reviewer rejects twice" path) without requiring
anyone to hand it an API key just to read the code. Set ANTHROPIC_API_KEY
and pass --mock=False (the default when a key is present) to use the real
model instead.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str


class BaseAgentLLM:
    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class AnthropicAgentLLM(BaseAgentLLM):
    """Real model calls via langchain-anthropic."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        from langchain_anthropic import ChatAnthropic

        self._client = ChatAnthropic(model=model, temperature=0.4, max_tokens=1024)

    def complete(self, system: str, user: str) -> LLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        result = self._client.invoke(messages)
        return LLMResponse(text=result.content)


class MockAgentLLM(BaseAgentLLM):
    """
    Deterministic-ish stand-in used for offline demos and tests.

    It doesn't call any API. Instead each "role" has a small hand-written
    behavior so the graph's control flow (the interesting part of this
    project) can be exercised and unit-tested without network access or
    API costs. The one bit of controllable randomness is the reviewer's
    verdict, which callers can force via `force_reject`.
    """

    def __init__(self, force_reject: Optional[bool] = None, seed: int = 7):
        self.force_reject = force_reject
        self._rng = random.Random(seed)

    def complete(self, system: str, user: str) -> LLMResponse:
        role = _infer_role(system)
        if role == "planner":
            return LLMResponse(text=_mock_plan(user))
        if role == "worker":
            return LLMResponse(text=_mock_draft(user))
        if role == "reviewer":
            return LLMResponse(text=_mock_review(user, self.force_reject, self._rng))
        return LLMResponse(text="(mock) no response")


def _infer_role(system: str) -> str:
    # Match on "you are the X agent" specifically, rather than "does the
    # word X appear anywhere" -- each system prompt mentions the *other*
    # agents by name too (e.g. the reviewer's prompt tells it to write
    # feedback "a WORKER agent can act on"), so a loose substring check
    # picks up the wrong role.
    system_lower = system.lower()
    for role in ("planner", "worker", "reviewer"):
        if f"you are the {role}" in system_lower:
            return role
    return "unknown"


def _mock_plan(user: str) -> str:
    return (
        "1. Restate the task in one sentence.\n"
        "2. List the 3 key points the answer must cover.\n"
        "3. Draft a short, direct response covering all 3 points.\n"
        "4. Keep it under 120 words."
    )


def _mock_draft(user: str) -> str:
    # Deliberately a bit thin on the first pass so the mock reviewer has
    # something legitimate to push back on -- mirrors what actually
    # happens with real models more often than a perfect-first-try demo.
    if "REVISION FEEDBACK" in user:
        return (
            "Revised draft: addresses the feedback directly, adds the missing "
            "detail, and keeps the response under 120 words as required by the plan."
        )
    return "Draft: a short answer to the task, following the plan's outline."


def _mock_review(user: str, force_reject: Optional[bool], rng: random.Random) -> str:
    is_first_pass = "this draft is a revision after prior feedback" not in user
    if force_reject is True:
        decision = "REJECT"
    elif force_reject is False:
        decision = "APPROVE"
    elif is_first_pass:
        # First pass: reject to demonstrate the send-back edge by default.
        decision = "REJECT"
    else:
        decision = "APPROVE"

    if decision == "REJECT":
        return "REJECT\nFeedback: the draft is missing a concrete detail and needs to be more specific."
    return "APPROVE\nFeedback: meets the plan's requirements."


def get_llm(mock: bool = True, model: str = "claude-sonnet-4-6", force_reject: Optional[bool] = None) -> BaseAgentLLM:
    if not mock and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicAgentLLM(model=model)
    return MockAgentLLM(force_reject=force_reject)
