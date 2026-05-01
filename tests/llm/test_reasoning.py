from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from synapsekit.llm.providers.anthropic_thinking import AnthropicThinking
from synapsekit.llm.providers.deepseek_r1 import DeepSeekR1Reasoning
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


# ---------------------------------------------------------------------------
# OpenAI provider — streaming and _reasoning_effort branches
# ---------------------------------------------------------------------------


def test_openai_reasoning_effort_high() -> None:
    provider = OpenAIReasoning(model="o3", api_key="k", budget_tokens=4096)
    assert provider._reasoning_effort() == "high"


def test_openai_reasoning_effort_medium_default() -> None:
    provider = OpenAIReasoning(model="o3", api_key="k")
    assert provider._reasoning_effort() == "medium"


def test_openai_reasoning_effort_medium_explicit() -> None:
    provider = OpenAIReasoning(model="o3", api_key="k", budget_tokens=1000)
    assert provider._reasoning_effort() == "medium"


def test_openai_reasoning_effort_low_budget() -> None:
    provider = OpenAIReasoning(model="o3", api_key="k", budget_tokens=256)
    assert provider._reasoning_effort() == "low"


def test_openai_get_client_caches() -> None:
    import types

    mock_openai = types.ModuleType("openai")
    sentinel = object()
    mock_openai.AsyncOpenAI = lambda **k: sentinel

    with patch.dict(sys.modules, {"openai": mock_openai}):
        provider = OpenAIReasoning(model="o3", api_key="test-key")
        provider._client = None
        c1 = provider._get_client()
        c2 = provider._get_client()
    assert c1 is c2


def test_openai_get_client_missing_package() -> None:
    with patch.dict(sys.modules, {"openai": None}):
        provider = OpenAIReasoning(model="o3", api_key="k")
        provider._client = None
        with pytest.raises(ImportError, match="openai package required"):
            provider._get_client()


@pytest.mark.asyncio
async def test_openai_stream_yields_thinking_and_answer() -> None:
    class _FakeOpenAIStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning="think step", content=None)
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning=None, content="answer text")
                    )
                ]
            )
            yield SimpleNamespace(choices=[])  # empty choices — skipped

    create = AsyncMock(return_value=_FakeOpenAIStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client

    chunks = [c async for c in provider.stream("Q")]
    assert chunks[0].is_thinking is True
    assert chunks[0].text == "think step"
    assert chunks[1].is_thinking is False
    assert chunks[1].text == "answer text"


@pytest.mark.asyncio
async def test_openai_stream_thinking_disabled() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning="hidden", content="answer")
                    )
                ]
            )

    create = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = OpenAIReasoning(model="o3", api_key="k", thinking=False)
    provider._get_client = lambda: client

    chunks = [c async for c in provider.stream("Q")]
    assert all(not c.is_thinking for c in chunks)


@pytest.mark.asyncio
async def test_openai_stream_wraps_sdk_errors() -> None:
    create = AsyncMock(side_effect=ValueError("bad stream"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client

    with pytest.raises(RuntimeError, match="openai reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


# ---------------------------------------------------------------------------
# Anthropic provider — streaming paths
# ---------------------------------------------------------------------------


def test_anthropic_get_client_missing_package() -> None:
    with patch.dict(sys.modules, {"anthropic": None}):
        provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k")
        provider._client = None
        with pytest.raises(ImportError, match="anthropic package required"):
            provider._get_client()


@pytest.mark.asyncio
async def test_anthropic_generate_thinking_disabled() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hidden reasoning"),
            SimpleNamespace(type="text", text="final answer"),
        ],
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=6,
            thinking_tokens=0,
            total_tokens=0,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k", thinking=False)
    provider._get_client = lambda: client

    result = await provider.generate("Hi")
    assert result.thinking is None
    assert result.thinking_tokens == 0
    assert result.answer == "final answer"


@pytest.mark.asyncio
async def test_anthropic_stream_yields_chunks() -> None:
    class _FakeAnthropicStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(thinking="reasoning step", type="thinking_delta"),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="answer text", type="text_delta"),
            )
            yield SimpleNamespace(type="message_start")  # unrecognised type — skipped

    create = AsyncMock(return_value=_FakeAnthropicStream())
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k")
    provider._get_client = lambda: client

    chunks = [c async for c in provider.stream("Q")]
    assert chunks[0].is_thinking is True
    assert "reasoning step" in chunks[0].text
    assert chunks[1].is_thinking is False
    assert chunks[1].text == "answer text"


@pytest.mark.asyncio
async def test_anthropic_stream_thinking_disabled() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(thinking="hidden", type="thinking_delta"),
            )
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="answer", type="text_delta"),
            )

    create = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k", thinking=False)
    provider._get_client = lambda: client

    chunks = [c async for c in provider.stream("Q")]
    assert all(not c.is_thinking for c in chunks)


@pytest.mark.asyncio
async def test_anthropic_stream_wraps_sdk_errors() -> None:
    create = AsyncMock(side_effect=ValueError("bad stream"))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k")
    provider._get_client = lambda: client

    with pytest.raises(RuntimeError, match="anthropic reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


def test_anthropic_extract_content_blocks_non_list() -> None:
    from synapsekit.llm.providers.anthropic_thinking import _extract_content_blocks

    answer, thinking = _extract_content_blocks("not a list")
    assert answer == ""
    assert thinking is None


def test_anthropic_extract_thinking_text_dict() -> None:
    from synapsekit.llm.providers.anthropic_thinking import _extract_thinking_text

    delta = {"thinking": "chain of thought"}
    assert _extract_thinking_text(delta) == "chain of thought"


def test_anthropic_extract_answer_text_dict() -> None:
    from synapsekit.llm.providers.anthropic_thinking import _extract_answer_text

    delta = {"text": "final answer"}
    assert _extract_answer_text(delta) == "final answer"


def test_anthropic_attr_or_key_dict_branch() -> None:
    from synapsekit.llm.providers.anthropic_thinking import _extract_attr_or_key

    assert _extract_attr_or_key({"key": "value"}, "key") == "value"
    assert _extract_attr_or_key({"key": "value"}, "missing") is None


# ---------------------------------------------------------------------------
# DeepSeek provider — generate and stream
# ---------------------------------------------------------------------------


def test_deepseek_base_url_qwen() -> None:
    from synapsekit.llm.providers.deepseek_r1 import (
        _DEEPSEEK_BASE_URL,
        _QWEN_BASE_URL,
    )

    provider = DeepSeekR1Reasoning(model="qwq-32b", provider="qwen")
    assert provider._base_url() == _QWEN_BASE_URL

    provider2 = DeepSeekR1Reasoning(model="deepseek-r1")
    assert provider2._base_url() == _DEEPSEEK_BASE_URL


def test_deepseek_get_client_missing_package() -> None:
    with patch.dict(sys.modules, {"openai": None}):
        provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
        provider._client = None
        with pytest.raises(ImportError, match="openai package required"):
            provider._get_client()


@pytest.mark.asyncio
async def test_deepseek_generate_extracts_reasoning() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="answer text",
                    reasoning_content="thinking steps",
                )
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=5,
            total_tokens=20,
            reasoning_tokens=15,
            completion_tokens_details=None,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client

    result = await provider.generate("Explain transformers")
    assert result.answer == "answer text"
    assert result.thinking == "thinking steps"
    assert result.thinking_tokens == 15
    assert result.total_tokens == 20


@pytest.mark.asyncio
async def test_deepseek_generate_empty_response() -> None:
    response = SimpleNamespace(choices=[], usage=None)
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client

    result = await provider.generate("Hi")
    assert result.answer == ""
    assert result.thinking is None
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_deepseek_generate_thinking_disabled() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="answer",
                    reasoning_content="hidden",
                )
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=4,
            total_tokens=4,
            reasoning_tokens=0,
            completion_tokens_details=None,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k", thinking=False)
    provider._get_client = lambda: client

    result = await provider.generate("Hi")
    assert result.thinking is None
    assert result.thinking_tokens == 0


@pytest.mark.asyncio
async def test_deepseek_generate_wraps_sdk_errors() -> None:
    create = AsyncMock(side_effect=ValueError("bad response"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client

    with pytest.raises(RuntimeError, match="deepseek reasoning request failed"):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_deepseek_stream_yields_chunks() -> None:
    class _FakeDeepSeekStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content="think step", content=None
                        )
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            reasoning_content=None, content="answer text"
                        )
                    )
                ]
            )
            yield SimpleNamespace(choices=[])  # empty — skipped

    create = AsyncMock(return_value=_FakeDeepSeekStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client

    chunks = [c async for c in provider.stream("Q")]
    assert chunks[0].is_thinking is True
    assert chunks[0].text == "think step"
    assert chunks[1].is_thinking is False
    assert chunks[1].text == "answer text"


@pytest.mark.asyncio
async def test_deepseek_stream_wraps_sdk_errors() -> None:
    create = AsyncMock(side_effect=ValueError("bad stream"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client

    with pytest.raises(RuntimeError, match="deepseek reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


# ---------------------------------------------------------------------------
# Google provider — additional branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_generate_thinking_disabled() -> None:
    thinking_part = SimpleNamespace(text="hidden", thought=True)
    answer_part = SimpleNamespace(text="answer", thought=False)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[thinking_part, answer_part])
    )
    response = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            candidates_token_count=4,
            thoughts_token_count=10,
            total_token_count=14,
        ),
    )
    gen_mock = AsyncMock(return_value=response)
    model_mock = SimpleNamespace(generate_content_async=gen_mock)

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k", thinking=False)
    provider._get_model = lambda: model_mock

    result = await provider.generate("Hi")
    assert result.thinking is None
    assert result.thinking_tokens == 0


@pytest.mark.asyncio
async def test_google_generate_none_response() -> None:
    gen_mock = AsyncMock(return_value=None)
    model_mock = SimpleNamespace(generate_content_async=gen_mock)

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock

    result = await provider.generate("Hi")
    assert result.answer == ""
    assert result.thinking is None


@pytest.mark.asyncio
async def test_google_generate_wraps_sdk_errors() -> None:
    gen_mock = AsyncMock(side_effect=ValueError("api error"))
    model_mock = SimpleNamespace(generate_content_async=gen_mock)

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock

    with pytest.raises(RuntimeError, match="google reasoning request failed"):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_google_stream_fallback_no_candidates() -> None:
    fallback_chunk = SimpleNamespace(text="fallback answer")  # no candidates attr

    class _FallbackStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield fallback_chunk

    gen_mock = AsyncMock(return_value=_FallbackStream())
    model_mock = SimpleNamespace(generate_content_async=gen_mock)

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock

    chunks = [c async for c in provider.stream("Q")]
    assert len(chunks) == 1
    assert chunks[0].text == "fallback answer"
    assert chunks[0].is_thinking is False


@pytest.mark.asyncio
async def test_google_stream_thinking_disabled() -> None:
    thinking_chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="hidden thinking", thought=True)]
                )
            )
        ]
    )
    answer_chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="answer", thought=False)]
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

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k", thinking=False)
    provider._get_model = lambda: model_mock

    chunks = [c async for c in provider.stream("Q")]
    assert all(not c.is_thinking for c in chunks)


@pytest.mark.asyncio
async def test_google_stream_wraps_sdk_errors() -> None:
    gen_mock = AsyncMock(side_effect=ValueError("stream error"))
    model_mock = SimpleNamespace(generate_content_async=gen_mock)

    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock

    with pytest.raises(RuntimeError, match="google reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


def test_google_is_thinking_part_dict_branch() -> None:
    from synapsekit.llm.providers.google_thinking import _is_thinking_part

    assert _is_thinking_part({"thought": True}) is True
    assert _is_thinking_part({"isThought": True}) is True
    assert _is_thinking_part({"thought": False}) is False
    assert _is_thinking_part({}) is False


def test_google_extract_response_parts_no_candidates() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_response_parts

    response = SimpleNamespace(candidates=None, text="fallback text")
    answer, thinking = _extract_response_parts(response)
    assert answer == "fallback text"
    assert thinking is None


def test_google_extract_chunk_parts_no_candidates() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_chunk_parts

    chunk = SimpleNamespace(text="plain text")  # no candidates
    thinking, answer = _extract_chunk_parts(chunk)
    assert thinking == ""
    assert answer == "plain text"


# ---------------------------------------------------------------------------
# Remaining branch coverage — stream=None, delta=None, total_tokens=0, mid-stream errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_none_returns_empty() -> None:
    create = AsyncMock(return_value=None)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client
    chunks = [c async for c in provider.stream("Q")]
    assert chunks == []


@pytest.mark.asyncio
async def test_openai_stream_skips_null_delta() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=None)])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(reasoning=None, content="ans"))]
            )

    create = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client
    chunks = [c async for c in provider.stream("Q")]
    assert len(chunks) == 1
    assert chunks[0].text == "ans"


@pytest.mark.asyncio
async def test_openai_stream_mid_iteration_error() -> None:
    class _ErrorStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(reasoning=None, content="a"))]
            )
            raise ValueError("mid-stream failure")

    create = AsyncMock(return_value=_ErrorStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = OpenAIReasoning(model="o3", api_key="k")
    provider._get_client = lambda: client
    with pytest.raises(RuntimeError, match="openai reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


@pytest.mark.asyncio
async def test_anthropic_stream_none_returns_empty() -> None:
    create = AsyncMock(return_value=None)
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k")
    provider._get_client = lambda: client
    chunks = [c async for c in provider.stream("Q")]
    assert chunks == []


@pytest.mark.asyncio
async def test_anthropic_stream_mid_iteration_error() -> None:
    class _ErrorStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="first", type="text_delta"),
            )
            raise ValueError("mid-stream failure")

    create = AsyncMock(return_value=_ErrorStream())
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="k")
    provider._get_client = lambda: client
    with pytest.raises(RuntimeError, match="anthropic reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


@pytest.mark.asyncio
async def test_anthropic_get_client_caches() -> None:
    import types

    mock_anthropic = types.ModuleType("anthropic")
    sentinel = object()
    mock_anthropic.AsyncAnthropic = lambda **k: sentinel

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        provider = AnthropicThinking(model="claude-3-7-sonnet", api_key="test-key")
        provider._client = None
        c1 = provider._get_client()
        c2 = provider._get_client()
    assert c1 is c2


def test_anthropic_extract_thinking_text_inner_dict() -> None:
    from synapsekit.llm.providers.anthropic_thinking import _extract_thinking_text

    # dict value where inner key "thinking" holds the text
    delta = SimpleNamespace(thinking={"thinking": "chain of thought"})
    result = _extract_thinking_text(delta)
    assert result == "chain of thought"


def test_anthropic_extract_answer_text_inner_dict() -> None:
    from synapsekit.llm.providers.anthropic_thinking import _extract_answer_text

    delta = SimpleNamespace(text={"text": "final answer"})
    result = _extract_answer_text(delta)
    assert result == "final answer"


@pytest.mark.asyncio
async def test_deepseek_stream_none_returns_empty() -> None:
    create = AsyncMock(return_value=None)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client
    chunks = [c async for c in provider.stream("Q")]
    assert chunks == []


@pytest.mark.asyncio
async def test_deepseek_stream_skips_null_delta() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=None)])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="ans"))]
            )

    create = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client
    chunks = [c async for c in provider.stream("Q")]
    assert len(chunks) == 1
    assert chunks[0].text == "ans"


@pytest.mark.asyncio
async def test_deepseek_stream_mid_iteration_error() -> None:
    class _ErrorStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="a"))]
            )
            raise ValueError("mid-stream failure")

    create = AsyncMock(return_value=_ErrorStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client
    with pytest.raises(RuntimeError, match="deepseek reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


@pytest.mark.asyncio
async def test_deepseek_generate_total_tokens_accumulation() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer", reasoning_content="think")
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=5,
            total_tokens=0,  # forces accumulation path
            reasoning_tokens=10,
            completion_tokens_details=None,
        ),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k")
    provider._get_client = lambda: client
    result = await provider.generate("Hi")
    assert result.total_tokens == 15  # 10 + 5


@pytest.mark.asyncio
async def test_google_stream_none_returns_empty() -> None:
    gen_mock = AsyncMock(return_value=None)
    model_mock = SimpleNamespace(generate_content_async=gen_mock)
    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock
    chunks = [c async for c in provider.stream("Q")]
    assert chunks == []


@pytest.mark.asyncio
async def test_google_stream_mid_iteration_error() -> None:
    class _ErrorStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="first", thought=False)]
                        )
                    )
                ]
            )
            raise ValueError("mid-stream failure")

    gen_mock = AsyncMock(return_value=_ErrorStream())
    model_mock = SimpleNamespace(generate_content_async=gen_mock)
    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock
    with pytest.raises(RuntimeError, match="google reasoning request failed"):
        async for _ in provider.stream("Q"):
            pass


@pytest.mark.asyncio
async def test_google_generate_total_tokens_accumulation() -> None:
    thinking_part = SimpleNamespace(text="think", thought=True)
    answer_part = SimpleNamespace(text="ans", thought=False)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[thinking_part, answer_part])
    )
    response = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            candidates_token_count=3,
            thoughts_token_count=7,
            total_token_count=0,  # forces accumulation path
        ),
    )
    gen_mock = AsyncMock(return_value=response)
    model_mock = SimpleNamespace(generate_content_async=gen_mock)
    provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
    provider._get_model = lambda: model_mock
    result = await provider.generate("Hi")
    assert result.total_tokens == 10  # 7 + 3


def test_google_extract_response_parts_empty_parts() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_response_parts

    # Part with no text — should be skipped via `if not part_text: continue`
    empty_part = SimpleNamespace(text="", thought=False)
    thinking_part = SimpleNamespace(text="think", thought=True)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[empty_part, thinking_part])
    )
    response = SimpleNamespace(candidates=[candidate])
    answer, thinking = _extract_response_parts(response)
    assert thinking == "think"
    assert answer == ""


def test_google_extract_chunk_parts_with_thinking() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_chunk_parts

    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="think", thought=True),
                        SimpleNamespace(text="ans", thought=False),
                    ]
                )
            )
        ]
    )
    thinking, answer = _extract_chunk_parts(chunk)
    assert thinking == "think"
    assert answer == "ans"


def test_google_get_model_missing_package() -> None:
    import types

    with patch.dict(sys.modules, {"google": None, "google.generativeai": None}):
        provider = GoogleThinking(model="gemini-2.5-pro", api_key="k")
        provider._model = None
        with pytest.raises(ImportError, match="google-generativeai required"):
            provider._get_model()


@pytest.mark.asyncio
async def test_deepseek_stream_thinking_disabled() -> None:
    class _FakeStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content="hidden", content="answer")
                )]
            )

    create = AsyncMock(return_value=_FakeStream())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="k", thinking=False)
    provider._get_client = lambda: client
    chunks = [c async for c in provider.stream("Q")]
    assert all(not c.is_thinking for c in chunks)


def test_deepseek_get_client_caches() -> None:
    import types

    mock_openai = types.ModuleType("openai")
    sentinel = object()
    mock_openai.AsyncOpenAI = lambda **k: sentinel

    with patch.dict(sys.modules, {"openai": mock_openai}):
        provider = DeepSeekR1Reasoning(model="deepseek-r1", api_key="test-key")
        provider._client = None
        c1 = provider._get_client()
        c2 = provider._get_client()
    assert c1 is c2


def test_google_extract_response_parts_no_parts_in_candidate() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_response_parts

    # Candidate with content=None → parts=None → hits `if not parts: continue`
    candidate = SimpleNamespace(content=None)
    response = SimpleNamespace(candidates=[candidate])
    answer, thinking = _extract_response_parts(response)
    assert answer == ""
    assert thinking is None


def test_google_extract_response_parts_skips_empty_part_text() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_response_parts

    empty_part = SimpleNamespace(text="", thought=False)
    real_part = SimpleNamespace(text="answer", thought=False)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[empty_part, real_part])
    )
    response = SimpleNamespace(candidates=[candidate])
    answer, thinking = _extract_response_parts(response)
    assert answer == "answer"
    assert thinking is None


def test_google_extract_chunk_parts_none() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_chunk_parts

    thinking, answer = _extract_chunk_parts(None)
    assert thinking == ""
    assert answer == ""


def test_google_extract_chunk_parts_no_parts_in_candidate() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_chunk_parts

    chunk = SimpleNamespace(candidates=[SimpleNamespace(content=None)])
    thinking, answer = _extract_chunk_parts(chunk)
    assert thinking == ""
    assert answer == ""


def test_google_extract_chunk_parts_skips_empty_part_text() -> None:
    from synapsekit.llm.providers.google_thinking import _extract_chunk_parts

    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="", thought=False),
                        SimpleNamespace(text="real", thought=False),
                    ]
                )
            )
        ]
    )
    thinking, answer = _extract_chunk_parts(chunk)
    assert thinking == ""
    assert answer == "real"
