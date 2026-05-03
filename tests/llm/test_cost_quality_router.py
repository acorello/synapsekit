"""Tests for CostQualityRouter — learning-based explore/exploit routing."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.cost_quality_router import CostQualityRouter


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class MockLLM(BaseLLM):
    """Controllable LLM stub — overrides generate() and stream() directly."""

    def __init__(
        self,
        model_name: str,
        response: str = "ok",
        input_tokens: int = 0,
        output_tokens: int = 0,
        fail: bool = False,
    ) -> None:
        super().__init__(LLMConfig(model=model_name, api_key="", provider="openai"))
        self._mock_response = response
        self._mock_input = input_tokens
        self._mock_output = output_tokens
        self._mock_fail = fail
        self.call_count = 0

    async def generate(self, prompt: str, **kw: Any) -> str:
        if self._mock_fail:
            raise RuntimeError(f"{self.config.model} failed")
        self.call_count += 1
        self._input_tokens += self._mock_input
        self._output_tokens += self._mock_output
        return self._mock_response

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        if self._mock_fail:
            raise RuntimeError(f"{self.config.model} failed")
        self.call_count += 1
        self._input_tokens += self._mock_input
        self._output_tokens += self._mock_output
        yield self._mock_response


def _pre_fill(router: CostQualityRouter, model: str, calls: int, avg_cost: float, avg_quality: float) -> None:
    """Manually populate router stats to simulate completed exploration."""
    router._stats[model] = {
        "calls": calls,
        "avg_cost": avg_cost,
        "avg_quality": avg_quality,
        "_total_quality": avg_quality * calls,
        "_quality_calls": calls,
    }


# ------------------------------------------------------------------ #
# 1. Exploration phase
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_exploration_uses_all_candidates():
    """All candidates are used during the exploration phase (round-robin)."""
    llm_a = MockLLM("model-a")
    llm_b = MockLLM("model-b")
    llm_c = MockLLM("model-c")
    router = CostQualityRouter(candidates=[llm_a, llm_b, llm_c], explore_n=9)

    for _ in range(9):  # all within explore window (_calls 1-9 <= 9)
        await router.call("prompt")

    assert router._mode == "explore"
    assert llm_a.call_count > 0
    assert llm_b.call_count > 0
    assert llm_c.call_count > 0


@pytest.mark.asyncio
async def test_exploration_round_robin_order():
    """Exploration cycles through candidates in round-robin order."""
    llm_a = MockLLM("model-a")
    llm_b = MockLLM("model-b")
    router = CostQualityRouter(candidates=[llm_a, llm_b], explore_n=100)

    for _ in range(6):
        await router.call("prompt")

    assert llm_a.call_count == 3
    assert llm_b.call_count == 3


@pytest.mark.asyncio
async def test_exploration_stats_updated():
    """Stats are updated for each call during exploration."""
    llm = MockLLM("model-a", input_tokens=100, output_tokens=50)
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    await router.call("prompt")
    await router.call("prompt")

    s = router._stats["model-a"]
    assert s["calls"] == 2
    # model-a not in COST_TABLE → cost stays 0.0
    assert s["avg_cost"] == 0.0


@pytest.mark.asyncio
async def test_exploration_mode_flag():
    """_mode is 'explore' during exploration and 'exploit' after."""
    llm_a = MockLLM("model-a")
    llm_b = MockLLM("model-b")
    router = CostQualityRouter(candidates=[llm_a, llm_b], explore_n=3)

    await router.call("p")
    assert router._mode == "explore"

    await router.call("p")
    await router.call("p")
    assert router._mode == "explore"

    await router.call("p")  # call 4 > explore_n=3 → exploit
    assert router._mode == "exploit"


# ------------------------------------------------------------------ #
# 2. Exploitation — cheapest valid model selected
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_exploitation_selects_cheapest_model():
    """Exploit mode picks the cheapest model that meets the quality threshold."""
    llm_cheap = MockLLM("gpt-4o-mini")
    llm_expensive = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_cheap, llm_expensive],
        quality_threshold=0.80,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.85)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.01, avg_quality=0.95)

    await router.call("prompt")

    assert llm_cheap.call_count == 1
    assert llm_expensive.call_count == 0


@pytest.mark.asyncio
async def test_exploitation_prefers_lower_cost_among_equal_quality():
    """Among models with equal quality, the cheaper one is preferred."""
    llm_a = MockLLM("gpt-4o-mini")
    llm_b = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_a, llm_b],
        quality_threshold=0.80,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.90)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.05, avg_quality=0.90)

    await router.call("prompt")

    assert llm_a.call_count == 1
    assert llm_b.call_count == 0


# ------------------------------------------------------------------ #
# 3. Quality threshold enforcement
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_threshold_excludes_low_quality_models():
    """Models below quality_threshold are not selected during exploitation."""
    llm_high = MockLLM("gpt-4o")
    llm_low = MockLLM("gpt-4o-mini")
    router = CostQualityRouter(
        candidates=[llm_high, llm_low],
        quality_threshold=0.90,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.01, avg_quality=0.95)
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.70)

    await router.call("prompt")

    assert llm_high.call_count == 1
    assert llm_low.call_count == 0


@pytest.mark.asyncio
async def test_threshold_fallback_when_none_qualify():
    """When no model meets the threshold, router falls back to highest-quality model."""
    llm_best = MockLLM("model-best")
    llm_worst = MockLLM("model-worst")
    router = CostQualityRouter(
        candidates=[llm_best, llm_worst],
        quality_threshold=0.99,  # impossible threshold
        explore_n=0,
    )
    _pre_fill(router, "model-best", 5, avg_cost=0.01, avg_quality=0.85)
    _pre_fill(router, "model-worst", 5, avg_cost=0.001, avg_quality=0.50)

    await router.call("prompt")

    assert llm_best.call_count == 1
    assert llm_worst.call_count == 0


@pytest.mark.asyncio
async def test_threshold_zero_accepts_all_models():
    """quality_threshold=0.0 accepts any model (cheapest wins)."""
    llm_cheap = MockLLM("gpt-4o-mini")
    llm_expensive = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_cheap, llm_expensive],
        quality_threshold=0.0,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.0)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.05, avg_quality=0.0)

    await router.call("prompt")

    assert llm_cheap.call_count == 1
    assert llm_expensive.call_count == 0


# ------------------------------------------------------------------ #
# 4. Fallback on failure
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_fallback_on_primary_failure_explore():
    """When the selected model fails during exploration, router retries with the next one."""
    llm_fail = MockLLM("model-fail", fail=True)
    llm_ok = MockLLM("model-ok")
    router = CostQualityRouter(candidates=[llm_fail, llm_ok], explore_n=100)

    result = await router.call("prompt")

    assert result == "ok"
    assert llm_ok.call_count == 1


@pytest.mark.asyncio
async def test_fallback_on_primary_failure_exploit():
    """When exploit-selected model fails, router falls back to the next candidate."""
    llm_fail = MockLLM("gpt-4o-mini", fail=True)
    llm_ok = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_fail, llm_ok],
        quality_threshold=0.0,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.90)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.01, avg_quality=0.95)

    result = await router.call("prompt")

    assert result == "ok"
    assert llm_ok.call_count == 1


@pytest.mark.asyncio
async def test_raises_when_all_candidates_fail():
    """RuntimeError propagates when every candidate fails."""
    llm_a = MockLLM("model-a", fail=True)
    llm_b = MockLLM("model-b", fail=True)
    router = CostQualityRouter(candidates=[llm_a, llm_b], explore_n=100)

    with pytest.raises(RuntimeError):
        await router.call("prompt")


@pytest.mark.asyncio
async def test_fallback_does_not_update_stats_for_failed_call():
    """Failed model calls are not counted in stats."""
    llm_fail = MockLLM("model-fail", fail=True)
    llm_ok = MockLLM("model-ok")
    router = CostQualityRouter(candidates=[llm_fail, llm_ok], explore_n=100)

    await router.call("prompt")

    assert router._stats["model-fail"]["calls"] == 0
    assert router._stats["model-ok"]["calls"] == 1


# ------------------------------------------------------------------ #
# 5. Budget constraint
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_budget_skips_over_budget_model():
    """Models whose avg_cost exceeds budget_per_call_usd are skipped."""
    llm_cheap = MockLLM("gpt-4o-mini")
    llm_expensive = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_cheap, llm_expensive],
        quality_threshold=0.80,
        budget_per_call_usd=0.001,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.85)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.05, avg_quality=0.95)

    await router.call("prompt")

    assert llm_cheap.call_count == 1
    assert llm_expensive.call_count == 0


@pytest.mark.asyncio
async def test_budget_uses_over_budget_as_fallback_when_necessary():
    """When all quality-meeting models exceed budget, the cheapest of them is used anyway."""
    llm_a = MockLLM("gpt-4o-mini")
    llm_b = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_a, llm_b],
        quality_threshold=0.80,
        budget_per_call_usd=0.000001,  # impossibly tight
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.001, avg_quality=0.85)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.05, avg_quality=0.95)

    result = await router.call("prompt")

    # Router should not crash; cheaper over-budget model is used as fallback
    assert result == "ok"
    assert llm_a.call_count + llm_b.call_count == 1


@pytest.mark.asyncio
async def test_budget_none_ignores_cost_constraint():
    """When budget_per_call_usd is None, no budget filtering is applied."""
    llm_a = MockLLM("gpt-4o-mini")
    llm_b = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_a, llm_b],
        quality_threshold=0.80,
        budget_per_call_usd=None,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.85)
    _pre_fill(router, "gpt-4o", 5, avg_cost=999.0, avg_quality=0.95)

    await router.call("prompt")

    # Without budget constraint, cheapest within threshold wins
    assert llm_a.call_count == 1
    assert llm_b.call_count == 0


# ------------------------------------------------------------------ #
# 6. Pareto frontier correctness
# ------------------------------------------------------------------ #


def test_pareto_frontier_excludes_dominated_model():
    """Model dominated by another (cheaper AND higher quality) is not on frontier."""
    llm_a = MockLLM("model-a")  # dominated
    llm_b = MockLLM("model-b")  # cheaper, higher quality than a
    llm_c = MockLLM("model-c")  # high quality, but more expensive than b
    router = CostQualityRouter(candidates=[llm_a, llm_b, llm_c], explore_n=100)

    # model-a: cost=0.10, quality=0.70 — dominated by model-b (cheaper, higher quality)
    _pre_fill(router, "model-a", 3, avg_cost=0.10, avg_quality=0.70)
    _pre_fill(router, "model-b", 3, avg_cost=0.01, avg_quality=0.80)
    _pre_fill(router, "model-c", 3, avg_cost=0.05, avg_quality=0.95)

    result = router.stats()
    frontier_models = {f["model"] for f in result["frontier"]}

    assert "model-a" not in frontier_models
    assert "model-b" in frontier_models
    assert "model-c" in frontier_models


def test_pareto_frontier_all_on_frontier():
    """Models trading off cost vs quality are all on the Pareto frontier."""
    llm_a = MockLLM("model-a")  # cheap, low quality
    llm_b = MockLLM("model-b")  # expensive, high quality
    router = CostQualityRouter(candidates=[llm_a, llm_b], explore_n=100)

    _pre_fill(router, "model-a", 3, avg_cost=0.001, avg_quality=0.70)
    _pre_fill(router, "model-b", 3, avg_cost=0.10, avg_quality=0.95)

    result = router.stats()
    frontier_models = {f["model"] for f in result["frontier"]}

    assert "model-a" in frontier_models
    assert "model-b" in frontier_models


def test_pareto_frontier_empty_when_no_calls():
    """Frontier is empty before any calls have been made."""
    llm_a = MockLLM("model-a")
    llm_b = MockLLM("model-b")
    router = CostQualityRouter(candidates=[llm_a, llm_b], explore_n=10)

    result = router.stats()

    assert result["frontier"] == []


def test_pareto_frontier_structure():
    """stats() returns the documented schema."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], explore_n=100)
    _pre_fill(router, "model-a", 2, avg_cost=0.001, avg_quality=0.80)

    result = router.stats()

    assert "models" in result
    assert "frontier" in result
    assert "model-a" in result["models"]
    m = result["models"]["model-a"]
    assert set(m.keys()) == {"avg_cost", "avg_quality", "calls"}
    assert isinstance(result["frontier"], list)
    if result["frontier"]:
        f = result["frontier"][0]
        assert set(f.keys()) == {"model", "cost", "quality"}


def test_pareto_frontier_single_model_always_on_frontier():
    """A single model with calls is always on its own frontier."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], explore_n=100)
    _pre_fill(router, "model-a", 1, avg_cost=0.01, avg_quality=0.85)

    result = router.stats()

    assert len(result["frontier"]) == 1
    assert result["frontier"][0]["model"] == "model-a"


# ------------------------------------------------------------------ #
# 7. No eval suite
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_no_eval_suite_router_still_works():
    """Router operates normally without an eval suite; quality stays at 0.0."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite=None, explore_n=5)

    result = await router.call("prompt")

    assert result == "ok"
    s = router._stats["model-a"]
    assert s["calls"] == 1
    assert s["avg_quality"] == 0.0
    assert s["_quality_calls"] == 0


@pytest.mark.asyncio
async def test_no_eval_suite_does_not_update_quality():
    """avg_quality remains 0.0 across multiple calls when no eval suite is set."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite=None, explore_n=10)

    for _ in range(5):
        await router.call("prompt")

    assert router._stats["model-a"]["avg_quality"] == 0.0
    assert router._stats["model-a"]["calls"] == 5


# ------------------------------------------------------------------ #
# 8. Eval suite integration
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_eval_suite_updates_avg_quality():
    """Quality score from eval suite is reflected in model stats."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite="fake.suite", explore_n=10)

    mock_result = MagicMock()
    mock_result.mean_score = 0.85
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_result)
    router._evaluator = mock_evaluator
    router._evaluator_loaded = True

    await router.call("prompt")

    s = router._stats["model-a"]
    assert s["avg_quality"] == pytest.approx(0.85)
    assert s["_quality_calls"] == 1


@pytest.mark.asyncio
async def test_eval_suite_running_average():
    """avg_quality is correctly maintained as a running average across calls."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite="fake.suite", explore_n=10)

    scores = [0.80, 0.90, 1.00]
    call_idx = 0

    async def fake_evaluate(question, answer):
        nonlocal call_idx
        result = MagicMock()
        result.mean_score = scores[call_idx]
        call_idx += 1
        return result

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = fake_evaluate
    router._evaluator = mock_evaluator
    router._evaluator_loaded = True

    for _ in range(3):
        await router.call("prompt")

    expected = sum(scores) / len(scores)
    assert router._stats["model-a"]["avg_quality"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_eval_suite_error_does_not_crash_router():
    """If the eval suite raises an exception, quality is None and the call still succeeds."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite="bad.suite", explore_n=10)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(side_effect=RuntimeError("eval crashed"))
    router._evaluator = mock_evaluator
    router._evaluator_loaded = True

    result = await router.call("prompt")

    assert result == "ok"
    assert router._stats["model-a"]["avg_quality"] == 0.0


@pytest.mark.asyncio
async def test_eval_suite_dict_score_format():
    """Evaluator returning a dict with 'score' key is supported."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite="fake.suite", explore_n=10)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value={"score": 0.75})
    router._evaluator = mock_evaluator
    router._evaluator_loaded = True

    await router.call("prompt")

    assert router._stats["model-a"]["avg_quality"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_eval_suite_float_return_format():
    """Evaluator returning a plain float is supported."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], eval_suite="fake.suite", explore_n=10)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=0.92)
    router._evaluator = mock_evaluator
    router._evaluator_loaded = True

    await router.call("prompt")

    assert router._stats["model-a"]["avg_quality"] == pytest.approx(0.92)


# ------------------------------------------------------------------ #
# 9. Cost tracking
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_cost_tracked_from_cost_table():
    """Cost is calculated from COST_TABLE when model is known."""
    from synapsekit.observability.tracer import COST_TABLE

    model = "gpt-4o-mini"
    pricing = COST_TABLE[model]
    in_tokens, out_tokens = 1000, 500
    expected_cost = in_tokens * pricing["input"] + out_tokens * pricing["output"]

    llm = MockLLM(model, input_tokens=in_tokens, output_tokens=out_tokens)
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    await router.call("prompt")

    assert router._stats[model]["avg_cost"] == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_cost_zero_for_unknown_model():
    """Models not in COST_TABLE default to 0.0 cost."""
    llm = MockLLM("unknown-model-xyz", input_tokens=500, output_tokens=200)
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    await router.call("prompt")

    assert router._stats["unknown-model-xyz"]["avg_cost"] == 0.0


@pytest.mark.asyncio
async def test_cost_running_average():
    """avg_cost is a running mean across multiple calls."""
    from synapsekit.observability.tracer import COST_TABLE

    model = "gpt-4o-mini"
    pricing = COST_TABLE[model]
    in_tok, out_tok = 100, 50
    per_call = in_tok * pricing["input"] + out_tok * pricing["output"]

    llm = MockLLM(model, input_tokens=in_tok, output_tokens=out_tok)
    router = CostQualityRouter(candidates=[llm], explore_n=10)

    for _ in range(3):
        await router.call("prompt")

    assert router._stats[model]["avg_cost"] == pytest.approx(per_call)


# ------------------------------------------------------------------ #
# 10. Edge cases
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_single_candidate_always_used():
    """With one candidate the router always uses it."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    for _ in range(10):
        await router.call("prompt")

    assert llm.call_count == 10


@pytest.mark.asyncio
async def test_explore_n_zero_immediately_exploits():
    """explore_n=0 means the very first call is in exploit mode."""
    llm_cheap = MockLLM("gpt-4o-mini")
    llm_expensive = MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[llm_cheap, llm_expensive],
        quality_threshold=0.0,
        explore_n=0,
    )
    _pre_fill(router, "gpt-4o-mini", 5, avg_cost=0.0001, avg_quality=0.80)
    _pre_fill(router, "gpt-4o", 5, avg_cost=0.05, avg_quality=0.90)

    await router.call("prompt")

    assert router._mode == "exploit"
    assert llm_cheap.call_count == 1


@pytest.mark.asyncio
async def test_response_none_handled_gracefully():
    """Router coerces None response to empty string without crashing."""

    class NoneResponseLLM(BaseLLM):
        async def generate(self, prompt: str, **kw: Any) -> str:
            return None  # type: ignore[return-value]

        async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
            yield ""

    llm = NoneResponseLLM(LLMConfig(model="none-model", api_key="", provider="openai"))
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    result = await router.call("prompt")
    assert result == ""


@pytest.mark.asyncio
async def test_stream_yields_tokens_and_updates_stats():
    """stream() yields all tokens and updates stats after completion."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    collected = []
    async for token in router.stream("prompt"):
        collected.append(token)

    assert collected == ["ok"]
    assert router._stats["model-a"]["calls"] == 1


@pytest.mark.asyncio
async def test_call_method_delegates_to_generate():
    """call() is an alias for generate()."""
    llm = MockLLM("model-a")
    router = CostQualityRouter(candidates=[llm], explore_n=5)

    r1 = await router.call("prompt")
    r2 = await router.generate("prompt")

    assert r1 == r2 == "ok"
    assert llm.call_count == 2


# ------------------------------------------------------------------ #
# 11. Package import guard
# ------------------------------------------------------------------ #


def test_cost_quality_router_import_from_package():
    from synapsekit.llm import CostQualityRouter

    assert CostQualityRouter is not None
