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
