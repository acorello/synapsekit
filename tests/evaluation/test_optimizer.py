from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from synapsekit.evaluation.optimizer import PromptCandidate, PromptOptimizer, PromptVariantRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(response: str = "[]"):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


def _sync_case(score: float, cost_usd: float = 0.0):
    def fn():
        return {"score": score, "cost_usd": cost_usd}

    return fn


def _async_case(score: float, cost_usd: float = 0.0):
    async def fn():
        return {"score": score, "cost_usd": cost_usd}

    return fn


def _prompt_aware_case(score_map: dict[str, float]):
    """Eval case that accepts a prompt kwarg and looks up the expected score."""

    async def fn(prompt: str = ""):
        return {"score": score_map.get(prompt, 0.0), "cost_usd": 0.0}

    return fn


# ---------------------------------------------------------------------------
# PromptVariantRunner unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_calls_sync_case():
    cases = [("c1", _sync_case(0.8))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("test prompt")
    assert result.score == pytest.approx(0.8)
    assert result.text == "test prompt"


@pytest.mark.asyncio
async def test_runner_calls_async_case():
    cases = [("c1", _async_case(0.6))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("p")
    assert result.score == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_runner_injects_prompt_when_case_accepts_it():
    score_map = {"injected": 0.95}
    cases = [("c1", _prompt_aware_case(score_map))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("injected")
    assert result.score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_runner_averages_multiple_cases():
    cases = [("c1", _async_case(0.4)), ("c2", _async_case(0.8))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("p")
    assert result.score == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_runner_accumulates_cost():
    cases = [("c1", _async_case(0.5, cost_usd=0.02)), ("c2", _async_case(0.5, cost_usd=0.03))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("p")
    assert result.cost_usd == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_runner_uses_named_metric():
    async def fn():
        return {"faithfulness": 0.88, "score": 0.1}

    runner = PromptVariantRunner([("c1", fn)], "faithfulness")
    result = await runner.run_variant("p")
    assert result.score == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_runner_missing_metric_scores_zero():
    async def fn():
        return {"other": 0.9}

    runner = PromptVariantRunner([("c1", fn)], "faithfulness")
    result = await runner.run_variant("p")
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_runner_empty_cases_returns_zero_score():
    runner = PromptVariantRunner([], "score")
    result = await runner.run_variant("p")
    assert result.score == pytest.approx(0.0)
    assert result.cost_usd == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_runner_case_exception_counted_as_zero():
    async def bad_fn():
        raise RuntimeError("boom")

    cases = [("bad", bad_fn), ("good", _async_case(1.0))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("p")
    assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_runner_metadata_contains_per_case_scores():
    cases = [("c1", _async_case(0.3)), ("c2", _async_case(0.7))]
    runner = PromptVariantRunner(cases, "score")
    result = await runner.run_variant("p")
    assert result.metadata is not None
    assert result.metadata["scores"] == pytest.approx([0.3, 0.7])
    assert result.metadata["n_cases"] == 2


# ---------------------------------------------------------------------------
# PromptOptimizer — manual variants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_variants_are_used_directly():
    llm = _mock_llm()
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", variants=["v1", "v2"])
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.7))]

    best = await optimizer.run("base", "improve")

    llm.generate.assert_not_called()
    assert len(optimizer.candidates) == 2
    assert best.score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_manual_variants_ranked_highest_score_first():
    score_map = {"v1": 0.5, "v2": 0.9, "v3": 0.3}
    optimizer = PromptOptimizer(
        llm=_mock_llm(), eval_suite=".", metric="score", variants=["v1", "v2", "v3"]
    )
    optimizer._load_eval_cases = lambda: [("c1", _prompt_aware_case(score_map))]

    best = await optimizer.run("base", "improve")

    assert best.text == "v2"
    assert best.score == pytest.approx(0.9)
    scores = [c.score for c in optimizer.candidates]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# PromptOptimizer — LLM variant generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_variants_generated_when_no_variants_provided():
    json_response = '["variant one", "variant two", "variant three"]'
    llm = _mock_llm(json_response)
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", n_variants=3)
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.5))]

    await optimizer.run("base prompt", "make it better")

    llm.generate.assert_called_once()
    assert len(optimizer.candidates) == 3
    assert {c.text for c in optimizer.candidates} == {"variant one", "variant two", "variant three"}


@pytest.mark.asyncio
async def test_llm_variants_generation_prompt_contains_base_and_instructions():
    llm = _mock_llm('["v1"]')
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", n_variants=1)
    optimizer._load_eval_cases = lambda: []

    await optimizer.run("MY BASE PROMPT", "MY INSTRUCTIONS")

    call_args = llm.generate.call_args[0][0]
    assert "MY BASE PROMPT" in call_args
    assert "MY INSTRUCTIONS" in call_args


@pytest.mark.asyncio
async def test_llm_variants_parsed_from_markdown_fenced_json():
    json_response = '```json\n["alpha", "beta"]\n```'
    llm = _mock_llm(json_response)
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", n_variants=2)
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.5))]

    await optimizer.run("base", "improve")

    assert {c.text for c in optimizer.candidates} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_llm_generation_failure_falls_back_to_base_prompt():
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", n_variants=3)
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.6))]

    best = await optimizer.run("base prompt", "improve")

    assert len(optimizer.candidates) == 1
    assert best.text == "base prompt"


@pytest.mark.asyncio
async def test_llm_generation_invalid_json_falls_back_to_base_prompt():
    llm = _mock_llm("not valid json at all")
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", n_variants=2)
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.5))]

    best = await optimizer.run("base prompt", "improve")

    assert best.text == "base prompt"


@pytest.mark.asyncio
async def test_n_variants_caps_generated_list():
    json_response = '["a", "b", "c", "d", "e"]'
    llm = _mock_llm(json_response)
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", n_variants=3)
    optimizer._load_eval_cases = lambda: []

    await optimizer.run("base", "improve")

    assert len(optimizer.candidates) == 3


# ---------------------------------------------------------------------------
# PromptOptimizer — budget enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_cap_stops_after_threshold_exceeded():
    # Each variant costs $0.06; budget is $0.10
    # v1 → total $0.06 (< 0.10, continue)
    # v2 → total $0.12 (>= 0.10, stop)
    optimizer = PromptOptimizer(
        llm=_mock_llm(),
        eval_suite=".",
        metric="score",
        variants=["v1", "v2", "v3", "v4"],
        budget_usd=0.10,
    )
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.5, cost_usd=0.06))]

    await optimizer.run("base", "improve")

    assert len(optimizer.candidates) == 2


@pytest.mark.asyncio
async def test_no_budget_runs_all_variants():
    optimizer = PromptOptimizer(
        llm=_mock_llm(),
        eval_suite=".",
        metric="score",
        variants=["v1", "v2", "v3", "v4", "v5"],
        budget_usd=None,
    )
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.5, cost_usd=1.0))]

    await optimizer.run("base", "improve")

    assert len(optimizer.candidates) == 5


@pytest.mark.asyncio
async def test_partial_run_still_returns_best_so_far():
    score_map = {"v1": 0.9, "v2": 0.3}
    optimizer = PromptOptimizer(
        llm=_mock_llm(),
        eval_suite=".",
        metric="score",
        variants=["v1", "v2", "v3"],
        budget_usd=0.05,
    )
    optimizer._load_eval_cases = lambda: [
        ("c1", _prompt_aware_case(score_map)),
        ("cost_case", _async_case(0.0, cost_usd=0.06)),
    ]

    best = await optimizer.run("base", "improve")

    assert len(optimizer.candidates) == 1
    assert best.text == "v1"


# ---------------------------------------------------------------------------
# PromptOptimizer — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_eval_suite_all_candidates_score_zero():
    optimizer = PromptOptimizer(
        llm=_mock_llm(), eval_suite=".", metric="score", variants=["v1", "v2"]
    )
    optimizer._load_eval_cases = lambda: []

    best = await optimizer.run("base", "improve")

    assert best.score == pytest.approx(0.0)
    assert len(optimizer.candidates) == 2


@pytest.mark.asyncio
async def test_empty_variants_list_generates_via_llm():
    llm = _mock_llm('["generated"]')
    optimizer = PromptOptimizer(llm=llm, eval_suite=".", metric="score", variants=[])
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.7))]

    best = await optimizer.run("base", "improve")

    llm.generate.assert_called_once()
    assert best.text == "generated"


@pytest.mark.asyncio
async def test_candidates_attribute_populated_after_run():
    optimizer = PromptOptimizer(
        llm=_mock_llm(), eval_suite=".", metric="score", variants=["x", "y", "z"]
    )
    optimizer._load_eval_cases = lambda: [("c1", _async_case(0.5))]

    assert optimizer.candidates == []
    await optimizer.run("base", "improve")
    assert len(optimizer.candidates) == 3


@pytest.mark.asyncio
async def test_candidates_always_sorted_descending():
    score_map = {"a": 0.2, "b": 0.8, "c": 0.5}
    optimizer = PromptOptimizer(
        llm=_mock_llm(), eval_suite=".", metric="score", variants=["a", "b", "c"]
    )
    optimizer._load_eval_cases = lambda: [("c1", _prompt_aware_case(score_map))]

    await optimizer.run("base", "improve")

    scores = [c.score for c in optimizer.candidates]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_prompt_candidate_dataclass_fields():
    c = PromptCandidate(text="hello", score=0.75, cost_usd=0.01, metadata={"k": "v"})
    assert c.text == "hello"
    assert c.score == 0.75
    assert c.cost_usd == 0.01
    assert c.metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_prompt_candidate_metadata_defaults_to_none():
    c = PromptCandidate(text="x", score=0.0, cost_usd=0.0)
    assert c.metadata is None
