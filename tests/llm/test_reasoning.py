from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synapsekit.llm.providers.anthropic_thinking import AnthropicThinking
from synapsekit.llm.providers.google_thinking import GoogleThinking
from synapsekit.llm.providers.openai_reasoning import OpenAIReasoning
from synapsekit.llm.reasoning import ReasoningLLM, ReasoningResponse, ReasoningStreamChunk


def test_auto_detection_openai_provider() -> None:
    from synapsekit.llm.providers.openai_reasoning import OpenAIReasoning

    llm = ReasoningLLM("o3")
    assert isinstance(llm._provider, OpenAIReasoning)


@pytest.mark.asyncio
async def test_response_structure() -> None:
    llm = ReasoningLLM("deepseek-r1")

    class _Provider:
        async def generate(self, prompt: str) -> ReasoningResponse:
            del prompt
            return ReasoningResponse(
                answer="Final answer",
                thinking="Plan",
                thinking_tokens=12,
                answer_tokens=8,
                total_tokens=20,
                model="deepseek-r1",
                provider="deepseek",
            )

        async def stream(self, prompt: str):
            del prompt
            if False:
                yield

    llm._provider = _Provider()
    response = await llm.agenerate("What is 2+2?")

    assert isinstance(response, ReasoningResponse)
    assert response.answer == "Final answer"
    assert response.thinking == "Plan"
    assert response.total_tokens == 20


@pytest.mark.asyncio
async def test_streaming_order_thinking_before_answer() -> None:
    llm = ReasoningLLM("deepseek-r1")

    class _Provider:
        async def generate(self, prompt: str) -> ReasoningResponse:
            del prompt
            return ReasoningResponse(
                answer="A",
                thinking="T",
                thinking_tokens=0,
                answer_tokens=0,
                total_tokens=0,
                model="deepseek-r1",
                provider="deepseek",
            )

        async def stream(self, prompt: str):
            del prompt
            yield ReasoningStreamChunk(text="think-1", is_thinking=True)
            yield ReasoningStreamChunk(text="ans-1", is_thinking=False)
            yield ReasoningStreamChunk(text="think-2", is_thinking=True)
            yield ReasoningStreamChunk(text="ans-2", is_thinking=False)

    llm._provider = _Provider()
    chunks = [chunk async for chunk in llm.astream("Q")]

    assert [c.text for c in chunks] == ["think-1", "ans-1", "ans-2"]
    first_answer = next(i for i, c in enumerate(chunks) if not c.is_thinking)
    assert all(c.is_thinking for c in chunks[:first_answer])


@pytest.mark.asyncio
async def test_stream_is_not_buffered() -> None:
    yielded: list[str] = []

    class _Provider:
        async def stream(self, prompt: str):
            del prompt
            yield ReasoningStreamChunk("T1", True)
            yield ReasoningStreamChunk("A1", False)

    llm = ReasoningLLM("deepseek-r1")
    llm._provider = _Provider()

    async for chunk in llm.astream("Q"):
        yielded.append(chunk.text)
        break

    assert yielded == ["T1"]


@pytest.mark.asyncio
async def test_thinking_after_answer_dropped() -> None:
    class _Provider:
        async def stream(self, prompt: str):
            del prompt
            yield ReasoningStreamChunk("ans", False)
            yield ReasoningStreamChunk("late-think", True)

    llm = ReasoningLLM("deepseek-r1")
    llm._provider = _Provider()

    chunks = [c.text async for c in llm.astream("Q")]

    assert "late-think" not in chunks


@pytest.mark.asyncio
async def test_budget_tokens_passed_to_anthropic() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="reasoning"),
            SimpleNamespace(type="text", text="answer"),
        ],
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=6,
            thinking_tokens=3,
            total_tokens=13,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k", budget_tokens=512)
    provider._get_client = lambda: client

    _ = await provider.generate("Explain this")
    kwargs = create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 512}


@pytest.mark.asyncio
async def test_missing_thinking_is_none() -> None:
    message = SimpleNamespace(content="Only final answer")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            completion_tokens=5,
            prompt_tokens=3,
            total_tokens=8,
            completion_tokens_details=None,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client

    result = await provider.generate("Hi")
    assert result.thinking is None
    assert result.answer == "Only final answer"
    assert result.thinking_tokens == 0


@pytest.mark.asyncio
async def test_empty_response_handled() -> None:
    response = SimpleNamespace(choices=[], usage=None)
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client

    result = await provider.generate("Hi")
    assert result.answer == ""
    assert result.thinking is None
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_thinking_disabled_for_openai() -> None:
    message = SimpleNamespace(content="Final", reasoning="Hidden chain")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            completion_tokens=4,
            total_tokens=0,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=9),
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    llm = ReasoningLLM("o3", thinking=False)
    llm._provider._get_client = lambda: client

    result = await llm.agenerate("Hi")
    assert result.thinking is None
    assert result.thinking_tokens == 0
    assert create.call_args.kwargs["reasoning_effort"] == "low"


def test_unsupported_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported reasoning model"):
        ReasoningLLM("gpt-4o")


# ---------------------------------------------------------------------------
# Google provider tests (M3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_tokens_passed_to_google() -> None:
    thinking_part = SimpleNamespace(text="reasoning steps", thought=True)
    answer_part = SimpleNamespace(text="final answer", thought=False)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[thinking_part, answer_part])
    )
    response = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            candidates_token_count=8,
            thoughts_token_count=15,
            total_token_count=30,
        ),
    )
    gen_mock = AsyncMock(return_value=response)
    model_mock = SimpleNamespace(generate_content_async=gen_mock)

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k", budget_tokens=512)
    provider._get_model = lambda: model_mock

    result = await provider.generate("Explain attention")

    kwargs = gen_mock.call_args.kwargs
    assert kwargs["generation_config"] == {"thinkingConfig": {"thinkingBudget": 512}}
    assert result.thinking == "reasoning steps"
    assert result.answer == "final answer"
    assert result.thinking_tokens == 15
    assert result.answer_tokens == 8
    assert result.total_tokens == 30


@pytest.mark.asyncio
async def test_google_stream_splits_thinking_from_answer() -> None:
    thinking_chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="think step", thought=True)]
                )
            )
        ]
    )
    answer_chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="answer text", thought=False)]
                )
            )
        ]
    )

    class _FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield thinking_chunk
            yield answer_chunk

    gen_mock = AsyncMock(return_value=_FakeStream())
    model_mock = SimpleNamespace(generate_content_async=gen_mock)
    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock

    chunks = [c async for c in provider.stream("Q")]

    assert chunks[0].is_thinking is True
    assert chunks[0].text == "think step"
    assert chunks[1].is_thinking is False
    assert chunks[1].text == "answer text"


def test_anthropic_default_budget_applied_when_thinking_true() -> None:
    provider = AnthropicThinking(model="claude-3-7-sonnet", thinking=True)
    payload = provider._thinking_payload()
    assert payload is not None
    assert payload["type"] == "enabled"
    assert payload["budget_tokens"] > 0


def test_google_default_budget_applied_when_thinking_true() -> None:
    provider = GoogleThinking(model="gemini-2.5-pro", thinking=True)
    config = provider._generation_config()
    assert config is not None
    assert config["thinkingConfig"]["thinkingBudget"] > 0


def test_anthropic_thinking_false_sends_no_payload() -> None:
    provider = AnthropicThinking(model="claude-3-7-sonnet", thinking=False)
    assert provider._thinking_payload() is None


def test_google_thinking_false_sends_no_config() -> None:
    provider = GoogleThinking(model="gemini-2.5-pro", thinking=False)
    assert provider._generation_config() is None


# ---------------------------------------------------------------------------
# Error propagation tests (M4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_provider_wraps_sdk_errors() -> None:
    create = AsyncMock(side_effect=ValueError("bad response"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client

    with pytest.raises(RuntimeError, match="openai reasoning request failed"):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_anthropic_provider_wraps_sdk_errors() -> None:
    create = AsyncMock(side_effect=ValueError("bad response"))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k")
    provider._get_client = lambda: client

    with pytest.raises(RuntimeError, match="anthropic reasoning request failed"):
        await provider.generate("Hi")
