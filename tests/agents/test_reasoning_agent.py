from __future__ import annotations

import asyncio

import pytest

from synapsekit.agents.reasoning_agent import ReasoningAgent, ReasoningAgentConfig, ReasoningDecision
from synapsekit.llm.reasoning import ReasoningLLM, ReasoningResponse, ReasoningStreamChunk


class _FixedClassifier:
    def __init__(self, complexity: str):
        self._complexity = complexity

    async def classify(self, query: str) -> ReasoningDecision:
        del query
        return ReasoningDecision(complexity=self._complexity, reason="fixed")


class _FastLLM:
    def __init__(self, answer: str):
        self._answer = answer

    async def generate_with_messages(self, messages, **kw):
        del messages, kw
        return f"Thought: ok\nFinal Answer: {self._answer}"

    async def stream_with_messages(self, messages, **kw):
        del messages, kw
        yield f"Thought: ok\nFinal Answer: {self._answer}"


def _make_reasoning_llm(answer: str, thinking_tokens: int = 5, delay: float = 0.0) -> ReasoningLLM:
    llm = ReasoningLLM("deepseek-r1")

    class _Provider:
        async def generate(self, prompt: str) -> ReasoningResponse:
            del prompt
            if delay:
                await asyncio.sleep(delay)
            return ReasoningResponse(
                answer=f"Thought: ok\nFinal Answer: {answer}",
                thinking="reasoning",
                thinking_tokens=thinking_tokens,
                answer_tokens=10,
                total_tokens=thinking_tokens + 10,
                model="deepseek-r1",
                provider="deepseek",
            )

        async def stream(self, prompt: str):
            del prompt
            yield ReasoningStreamChunk(text="Final", is_thinking=False)

    llm._provider = _Provider()
    return llm


@pytest.mark.asyncio
async def test_routes_simple_to_fast():
    fast = _FastLLM("fast")
    reasoning = _make_reasoning_llm("reasoning")

    agent = ReasoningAgent(
        ReasoningAgentConfig(
            fast_llm=fast,
            reasoning_llm=reasoning,
            tools=[],
        )
    )
    agent._classifier = _FixedClassifier("simple")

    result = await agent.run("hi")

    assert result == "fast"


@pytest.mark.asyncio
async def test_routes_complex_to_reasoning():
    fast = _FastLLM("fast")
    reasoning = _make_reasoning_llm("reasoning")

    agent = ReasoningAgent(
        ReasoningAgentConfig(
            fast_llm=fast,
            reasoning_llm=reasoning,
            tools=[],
        )
    )
    agent._classifier = _FixedClassifier("complex")

    result = await agent.run("hard")

    assert result == "reasoning"
    assert agent.last_usage is not None


@pytest.mark.asyncio
async def test_budget_exceeded_falls_back():
    fast = _FastLLM("fast")
    reasoning = _make_reasoning_llm("reasoning", thinking_tokens=50)

    agent = ReasoningAgent(
        ReasoningAgentConfig(
            fast_llm=fast,
            reasoning_llm=reasoning,
            tools=[],
            thinking_budget_tokens=10,
        )
    )
    agent._classifier = _FixedClassifier("complex")

    result = await agent.run("hard")

    assert result == "fast"
    assert agent.last_fallback_reason is not None


@pytest.mark.asyncio
async def test_timeout_falls_back():
    fast = _FastLLM("fast")
    reasoning = _make_reasoning_llm("reasoning", delay=0.05)

    agent = ReasoningAgent(
        ReasoningAgentConfig(
            fast_llm=fast,
            reasoning_llm=reasoning,
            tools=[],
            timeout_seconds=0.01,
        )
    )
    agent._classifier = _FixedClassifier("complex")

    result = await agent.run("hard")

    assert result == "fast"
    assert agent.last_fallback_reason is not None
