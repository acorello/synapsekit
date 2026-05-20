"""Compatibility shims for evaluation exports used by the top-level package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class _AlertSink:
    def __init__(self, destination: str | None = None) -> None:
        self.destination = destination
        self.last_alert: dict[str, Any] | None = None

    def send(self, message: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"message": message, **kwargs}
        self.last_alert = payload
        return payload


class EmailAlertSink(_AlertSink):
    """Compatibility email alert sink."""


class PagerDutyAlertSink(_AlertSink):
    """Compatibility PagerDuty alert sink."""


class SlackWebhookAlertSink(_AlertSink):
    """Compatibility Slack webhook alert sink."""


class RAGAlertSink(_AlertSink):
    """Compatibility RAG alert sink."""


@dataclass(slots=True)
class RAGAlert:
    """Simple alert record for RAG evaluation."""

    message: str
    severity: str = "warning"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGERemediationSuggestion:
    """Suggested remediation for a failing RAG check."""

    message: str
    actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RAGEvaluationThresholds:
    """Thresholds used by the compatibility evaluator."""

    score: float = 0.8
    groundedness: float = 0.8
    faithfulness: float = 0.8


@dataclass(slots=True)
class RAGEvaluationResult:
    """Evaluation result summary."""

    score: float
    alerts: list[RAGAlert] = field(default_factory=list)
    suggestions: list[RAGERemediationSuggestion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """Tiny compatibility evaluator for the top-level export surface."""

    def __init__(self, thresholds: RAGEvaluationThresholds | None = None) -> None:
        self.thresholds = thresholds or RAGEvaluationThresholds()

    def evaluate(
        self, score: float, *, metadata: dict[str, Any] | None = None
    ) -> RAGEvaluationResult:
        alerts: list[RAGAlert] = []
        suggestions: list[RAGERemediationSuggestion] = []
        if score < self.thresholds.score:
            alerts.append(RAGAlert(message="score below threshold", severity="warning"))
            suggestions.append(
                RAGERemediationSuggestion(
                    message="Improve retrieval or answer grounding.",
                    actions=["review retrieved context", "tighten answer constraints"],
                )
            )
        return RAGEvaluationResult(
            score=score,
            alerts=alerts,
            suggestions=suggestions,
            metadata=metadata or {},
        )
