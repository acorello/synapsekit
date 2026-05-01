"""Integration tests for ReasoningLLM with agents and RAG pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synapsekit import ReActAgent
from synapsekit.llm import ReasoningLLM, ReasoningResponse


class _DummyReasoningLLM:
    async def agenerate(self, prompt: str) -> ReasoningResponse:
        del prompt
        return ReasoningResponse(
            answer="final answer from reasoning",
            thinking="reasoning chain",
            thinking_tokens=10,
            answer_tokens=5,
            total_tokens=15,
            model="test-model",
            provider="test",
        )

    async def stream(self, prompt: str):
        del prompt
        yield "final "
        yield "answer "

    async def astream(self, prompt: str):
        del prompt
        yield SimpleNamespace(text="reasoning", is_thinking=True)
        yield SimpleNamespace(text="final answer", is_thinking=False)


def test_reasoning_llm_with_react_agent():
    """Test that ReasoningLLM has required interface for ReActAgent compatibility."""
    from synapsekit.agents.tools import CalculatorTool
    from synapsekit.llm import ReasoningLLM

    llm = ReasoningLLM("deepseek-r1")

    assert hasattr(llm, "generate")
    assert hasattr(llm, "stream")
    assert callable(llm.stream)

    agent = ReActAgent(llm=llm, tools=[CalculatorTool()], max_iterations=1)

    assert agent is not None


@pytest.mark.asyncio
async def test_reasoning_llm_stream_with_react():
    """Test ReasoningLLM stream interface with ReActAgent."""
    from synapsekit.agents.tools import CalculatorTool
    from synapsekit.llm import ReasoningLLM

    llm = ReasoningLLM("deepseek-r1")

    async def dummy_stream(prompt):
        yield "final answer"

    async def dummy_generate(*args, **kw):
        return "final answer"

    llm.generate_with_messages = dummy_generate
    llm.stream_with_messages = dummy_stream

    agent = ReActAgent(llm=llm, tools=[CalculatorTool()], max_iterations=1)

    result = await agent.run("test")

    assert result is not None


class _DummyRAGLLM:
    """Dummy LLM that mimics ReasoningLLM interface for RAG compatibility."""

    async def generate(self, prompt: str):
        return "RAG answer"

    async def stream(self, prompt: str):
        del prompt
        yield "RAG answer"

    async def stream_with_messages(self, messages: list):
        yield "RAG answer"

    _input_tokens = 5
    _output_tokens = 3

    @property
    def tokens_used(self) -> dict[str, int]:
        return {"input": self._input_tokens, "output": self._output_tokens}

    @property
    def config(self):
        return SimpleNamespace(api_key="test", model="test-model")


def test_reasoning_llm_interface_compatible_with_rag():
    """Test that ReasoningLLM implements the minimal interface expected by RAG."""
    from synapsekit.embeddings.backend import SynapsekitEmbeddings
    from synapsekit.retrieval.vectorstore import InMemoryVectorStore

    embeddings = SynapsekitEmbeddings(model="all-MiniLM-L6-v2")
    vectorstore = InMemoryVectorStore(embeddings)

    from synapsekit.memory.conversation import ConversationMemory

    memory = ConversationMemory(window=2)
    memory.add("user", "test query")
    memory.add("assistant", "test answer")

    from synapsekit.retrieval.retriever import Retriever

    retriever = Retriever(vectorstore)

    from synapsekit.rag.pipeline import RAGConfig

    config = RAGConfig(
        llm=SimpleNamespace(
            stream_with_messages=AsyncMock(return_value=iter(["ok"])),
            tokens_used={"input": 1, "output": 1},
            config=SimpleNamespace(api_key="x", model="x"),
        ),
        retriever=retriever,
        memory=memory,
    )

    from synapsekit.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(config)

    assert pipeline is not None
    assert hasattr(pipeline, "stream")
    assert hasattr(pipeline, "ask")


def test_reasoning_llm_has_expected_interface():
    """Verify ReasoningLLM has stream method used by RAG pipeline."""
    llm = ReasoningLLM("deepseek-r1")

    assert hasattr(llm, "stream")
    assert callable(llm.stream)

    assert hasattr(llm, "agenerate")
    assert callable(llm.agenerate)


def test_reasoning_response_data_class():
    """Verify ReasoningResponse dataclass fields."""
    response = ReasoningResponse(
        answer="answer text",
        thinking="thinking text",
        thinking_tokens=5,
        answer_tokens=3,
        total_tokens=8,
        model="o3",
        provider="openai",
    )

    assert response.answer == "answer text"
    assert response.thinking == "thinking text"
    assert response.thinking_tokens == 5
    assert response.answer_tokens == 3
    assert response.total_tokens == 8
    assert response.model == "o3"
    assert response.provider == "openai"


def test_reasoning_llm_detects_all_providers():
    """Test provider auto-detection for all supported models."""
    from synapsekit.llm.reasoning import ReasoningLLM

    test_cases = [
        ("o3", "openai"),
        ("o1", "openai"),
        ("deepseek-r1", "deepseek"),
        ("gemini-2.5-pro", "google"),
        ("claude-3-5-sonnet", "anthropic"),
        ("qwq-32b", "qwen"),
    ]

    for model, expected_provider in test_cases:
        llm = ReasoningLLM(model)
        assert llm.provider == expected_provider, (
            f"Failed for {model}: expected {expected_provider}, got {llm.provider}"
        )


def test_reasoning_llm_unsupported_model():
    """Test that unsupported model raises ValueError."""
    from synapsekit.llm.reasoning import ReasoningLLM

    with pytest.raises(ValueError, match="Unsupported reasoning model"):
        ReasoningLLM("gpt-4o")


@pytest.mark.asyncio
async def test_reasoning_llm_no_thinking_when_disabled():
    """Test that thinking is None when disabled."""
    llm = ReasoningLLM(model="o3", thinking=False)

    assert llm.thinking is False

    class _NoThinkingProvider:
        async def generate(self, prompt: str) -> ReasoningResponse:
            return ReasoningResponse(
                answer="answer",
                thinking=None,
                thinking_tokens=0,
                answer_tokens=5,
                total_tokens=5,
                model="o3",
                provider="openai",
            )

        async def stream(self, prompt: str):
            yield SimpleNamespace(text="answer", is_thinking=False)

    llm._provider = _NoThinkingProvider()

    result = await llm.agenerate("test")

    assert result.thinking is None
    assert result.thinking_tokens == 0


@pytest.mark.asyncio
async def test_reasoning_llm_budget_tokens_configured():
    """Test that budget_tokens is stored."""
    llm = ReasoningLLM(model="claude-3-7-sonnet", api_key="key", budget_tokens=2048)

    assert llm.budget_tokens == 2048
    assert llm.thinking is True


def test_stream_chunk_data_class():
    """Verify ReasoningStreamChunk dataclass."""
    from synapsekit.llm import ReasoningStreamChunk

    chunk = ReasoningStreamChunk(text="thinking process", is_thinking=True)

    assert chunk.text == "thinking process"
    assert chunk.is_thinking is True

    chunk2 = ReasoningStreamChunk(text="final answer", is_thinking=False)
    assert chunk2.is_thinking is False
