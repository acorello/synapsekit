"""Behavioral tests for the sampled RAG evaluator."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from synapsekit.evaluation.rag_evaluator import RAGEvaluator
from synapsekit.observability.cost_tracker import CostTracker


class _JudgeLLM:
    def __init__(self, response: str, model: str = "gpt-4o-mini") -> None:
        self._response = response
        self.config = SimpleNamespace(model=model)
        self._input_tokens = 0
        self._output_tokens = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        self._input_tokens += 42
        self._output_tokens += 24
        return self._response

    @property
    def tokens_used(self) -> dict[str, int]:
        return {"input": self._input_tokens, "output": self._output_tokens}


class _RecordingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def send(self, alert, result) -> None:
        self.calls.append((alert, result))


@pytest.mark.asyncio
async def test_sampling_skips_judge_call(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = _JudgeLLM('{"recall": 1.0}')
    evaluator = RAGEvaluator(judge, sample_rate=0.5)
    monkeypatch.setattr(evaluator, "_sample_fraction", lambda _key: 0.99)

    result = await evaluator.evaluate(
        question="What is RAG?",
        answer="It is retrieval-augmented generation.",
        contexts=["retrieval-augmented generation is a pattern."],
        sample_key="skip-me",
    )

    assert result.sampled is False
    assert result.notes == "skipped by sampling"
    assert judge.prompts == []
    summary = evaluator.summary()
    assert summary["evaluations"] == 1
    assert summary["sampled_evaluations"] == 0
    assert summary["skipped_evaluations"] == 1


@pytest.mark.asyncio
async def test_evaluate_scores_alerts_and_cost_tracking() -> None:
    judge = _JudgeLLM(
        json.dumps(
            {
                "recall": 0.35,
                "precision": 0.20,
                "relevance": 0.25,
                "answer_quality": 0.40,
                "notes": "retrieval is noisy and the answer is incomplete",
            }
        )
    )
    cost_tracker = CostTracker()
    sink = _RecordingSink()
    evaluator = RAGEvaluator(judge, sample_rate=1.0, cost_tracker=cost_tracker, alert_sinks=[sink])

    result = await evaluator.evaluate(
        question="How does RAG help production answers?",
        answer="It helps a bit.",
        contexts=["chunk one", "chunk two"],
        sample_key="always-sample",
    )

    assert result.sampled is True
    assert result.recall == pytest.approx(0.35)
    assert result.precision == pytest.approx(0.20)
    assert result.relevance == pytest.approx(0.25)
    assert result.answer_quality == pytest.approx(0.40)
    assert result.retrieval_benefit == pytest.approx(0.325)
    assert result.benefit_to_cost is not None
    assert any(alert.metric == "precision" for alert in result.alerts)
    assert any("rerank" in suggestion.action.lower() for suggestion in result.suggestions)
    assert len(cost_tracker.records) == 1
    assert cost_tracker.total_cost_usd > 0
    assert sink.calls

    summary = evaluator.summary()
    assert summary["evaluations"] == 1
    assert summary["sampled_evaluations"] == 1
    assert summary["avg_precision"] == pytest.approx(0.20)
    assert summary["alerts"]["total"] >= 1
    assert summary["last_alerts"]
    assert summary["last_suggestions"]
    assert summary["last_question"] == "How does RAG help production answers?"


def test_sample_rate_lower_bound_is_enforced() -> None:
    judge = _JudgeLLM('{"recall": 1.0}')
    with pytest.raises(ValueError, match=r"0.01 and 1.0"):
        RAGEvaluator(judge, sample_rate=0.0)


def test_extract_json_object_handles_escaped_quotes():
    raw = 'Here is the result: {"notes": "The \\"quoted\\" term is here.", "recall": 0.8}'
    extracted = RAGEvaluator._extract_json_object(raw)
    parsed = json.loads(extracted)
    assert parsed["notes"] == 'The "quoted" term is here.'
    assert parsed["recall"] == 0.8


@pytest.mark.asyncio
async def test_benefit_to_cost_calculation():
    judge = _JudgeLLM('{"recall": 0.5, "precision": 0.5}')
    # Mocking cost: 1000 input tokens * $0.15/1M + 1000 output tokens * $0.60/1M = $0.00075
    # retrieval_benefit = (0.5 + 0.5) / 2 = 0.5
    # benefit_to_cost = 0.5 / 0.00075 = 666.666

    evaluator = RAGEvaluator(judge, sample_rate=1.0)

    # We need to ensure _estimate_eval_cost_usd returns a known value
    # gpt-4o-mini is in COST_TABLE in tracer.py
    # From synapsekit/observability/tracer.py (assumed):
    # "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6}

    result = await evaluator.evaluate("q", "a", ["c"])

    # _JudgeLLM uses 42 input, 24 output tokens
    # 42 * 0.15e-6 + 24 * 0.60e-6 = 0.0000063 + 0.0000144 = 0.0000207
    # benefit_to_cost = 0.5 / 0.0000207 = 24154.589

    assert result.eval_cost_usd == pytest.approx(0.0000207)
    assert result.benefit_to_cost == pytest.approx(0.5 / 0.0000207)


@pytest.mark.asyncio
async def test_alert_thresholds_logic():
    judge = _JudgeLLM('{"recall": 0.6, "precision": 0.8}')
    # thresholds.recall defaults to 0.65. 0.6 < 0.65 -> alert.

    evaluator = RAGEvaluator(judge, sample_rate=1.0)
    result = await evaluator.evaluate("q", "a", ["c"])

    recall_alerts = [a for a in result.alerts if a.metric == "recall"]
    assert len(recall_alerts) == 1
    assert recall_alerts[0].severity == "warning"  # 0.6 is > 0.65 * 0.5

    # Test critical severity
    judge._response = '{"recall": 0.3}'  # 0.3 < 0.65 * 0.5 = 0.325
    result = await evaluator.evaluate("q2", "a2", ["c2"])
    recall_alerts = [a for a in result.alerts if a.metric == "recall"]
    assert recall_alerts[0].severity == "critical"


@pytest.mark.asyncio
async def test_emit_alerts_swallows_sink_exceptions():
    judge = _JudgeLLM('{"recall": 0.1}')

    class FailingSink:
        async def send(self, alert, result):
            raise RuntimeError("Sink failed!")

    sink = FailingSink()
    # If this doesn't raise, then it's best-effort
    evaluator = RAGEvaluator(judge, sample_rate=1.0, alert_sinks=[sink])

    # Should not raise
    await evaluator.evaluate("q", "a", ["c"])
    assert len(evaluator.history) == 1
