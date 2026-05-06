from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from .base import BaseLLM, LLMConfig


@dataclass
class ReasoningResponse:
    answer: str
    thinking: str | None
    thinking_tokens: int
    answer_tokens: int
    total_tokens: int
    model: str
    provider: str


@dataclass
class ReasoningStreamChunk:
    text: str
    is_thinking: bool


class BaseReasoningProvider(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        budget_tokens: int | None = None,
        thinking: bool = True,
        provider: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.budget_tokens = budget_tokens
        self.thinking = thinking
        self.provider = provider or "unknown"

    @abstractmethod
    async def generate(self, prompt: str) -> ReasoningResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[ReasoningStreamChunk, None]:
        raise NotImplementedError

    def _empty_response(self) -> ReasoningResponse:
        return ReasoningResponse(
            answer="",
            thinking=None,
            thinking_tokens=0,
            answer_tokens=0,
            total_tokens=0,
            model=self.model,
            provider=self.provider,
        )


class ReasoningLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        budget_tokens: int | None = None,
        thinking: bool = True,
    ) -> None:
        provider = self._detect_provider(model)
        config = LLMConfig(
            model=model,
            api_key=api_key or "",
            provider=provider,
        )
        super().__init__(config)

        self.model = model
        self.provider = provider
        self.budget_tokens = budget_tokens
        self.thinking = thinking
        self._provider = self._build_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            budget_tokens=budget_tokens,
            thinking=thinking,
        )

    @staticmethod
    def _detect_provider(model: str) -> str:
        normalized_model = model.lower()
        if normalized_model.startswith(("o1", "o3")):
            return "openai"
        if "deepseek" in normalized_model:
            return "deepseek"
        if normalized_model.startswith("gemini"):
            return "google"
        if normalized_model.startswith("claude"):
            return "anthropic"
        if normalized_model.startswith("qwq"):
            return "qwen"
        raise ValueError("Unsupported reasoning model")

    def _build_provider(
        self,
        provider: str,
        model: str,
        api_key: str | None,
        budget_tokens: int | None,
        thinking: bool,
    ) -> BaseReasoningProvider:
        if provider == "openai":
            from .providers.openai_reasoning import OpenAIReasoning

            return OpenAIReasoning(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
            )
        if provider == "anthropic":
            from .providers.anthropic_thinking import AnthropicThinking

            return AnthropicThinking(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
            )
        if provider == "google":
            from .providers.google_thinking import GoogleThinking

            return GoogleThinking(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
            )
        if provider in {"deepseek", "qwen"}:
            from .providers.deepseek_r1 import DeepSeekR1Reasoning

            return DeepSeekR1Reasoning(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
                provider=provider,
            )
        raise ValueError("Unsupported reasoning model")

    async def agenerate(self, prompt: str) -> ReasoningResponse:
        return await self._provider.generate(prompt)

    async def astream(self, prompt: str) -> AsyncGenerator[ReasoningStreamChunk, None]:
        seen_answer = False
        async for chunk in self._provider.stream(prompt):
            if chunk.is_thinking:
                # Enforce reasoning-before-answer ordering.
                # Any thinking emitted after answer is ignored by design.
                if seen_answer:
                    continue
                yield chunk
                continue

            seen_answer = True
            yield chunk

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        del kw
        async for chunk in self.astream(prompt):
            if not chunk.is_thinking:
                yield chunk.text
