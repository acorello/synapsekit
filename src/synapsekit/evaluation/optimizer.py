"""Prompt optimization helpers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(slots=True)
class PromptCandidate:
    """A candidate prompt variant."""

    name: str
    prompt: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.prompt = str(self.prompt)
        if self.score is not None:
            self.score = float(self.score)


class PromptVariantRunner:
    """Run prompt variants through a provided callable."""

    def __init__(self, runner: Callable[[str], Any] | None = None) -> None:
        self._runner = runner

    async def run(self, candidate: PromptCandidate) -> Any:
        if self._runner is None:
            return candidate.prompt
        return await _maybe_await(self._runner(candidate.prompt))

    async def run_all(self, candidates: Sequence[PromptCandidate]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            results.append(
                {
                    "candidate": candidate,
                    "result": await self.run(candidate),
                }
            )
        return results


class PromptOptimizer:
    """Select the best prompt variant based on a scoring function."""

    def __init__(
        self,
        scorer: Callable[[str], float | Awaitable[float]] | None = None,
    ) -> None:
        self._scorer = scorer or (lambda prompt: len(prompt) / 100.0)

    async def score(self, prompt: str) -> float:
        value = await _maybe_await(self._scorer(prompt))
        return float(value)

    async def optimize(
        self,
        base_prompt: str,
        variants: Sequence[str | PromptCandidate] | None = None,
    ) -> PromptCandidate:
        candidates = self._build_candidates(base_prompt, variants)
        best = candidates[0]
        best.score = await self.score(best.prompt)

        for candidate in candidates[1:]:
            candidate.score = await self.score(candidate.prompt)
            if candidate.score > (best.score or float("-inf")):
                best = candidate

        return best

    def _build_candidates(
        self,
        base_prompt: str,
        variants: Sequence[str | PromptCandidate] | None,
    ) -> list[PromptCandidate]:
        if not variants:
            return [PromptCandidate(name="base", prompt=base_prompt)]

        candidates: list[PromptCandidate] = []
        for index, variant in enumerate(variants):
            if isinstance(variant, PromptCandidate):
                candidates.append(variant)
            else:
                candidates.append(PromptCandidate(name=f"variant-{index + 1}", prompt=str(variant)))
        return candidates
