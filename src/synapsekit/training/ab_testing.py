"""A/B test router for comparing base vs fine-tuned model variants."""

from __future__ import annotations

import hashlib
import statistics
import threading
from collections import deque
from datetime import datetime
from typing import Literal

from .types import ABTestMetrics, ABTestResult


class ABTestRouter:
    """
    Routes production traffic between a base model and a fine-tuned variant.

    Sticky assignment ensures a given ``user_id`` always receives the same
    variant within an experiment — computed via a deterministic SHA-256
    bucket, so no session state is needed.  Rollout percentage is adjustable
    at runtime without invalidating past assignments for users below the new
    threshold.

    Parameters
    ----------
    base_model:
        Identifier of the current production model.
    finetuned_model:
        Identifier of the candidate fine-tuned model.
    rollout_pct:
        Percentage of traffic (0-100) sent to the fine-tuned model.
    experiment_id:
        Salt for the sticky hash.  Changing this resets all assignments,
        which is useful when starting a new experiment on the same models.
    max_results:
        Maximum number of ABTestResult records to retain in memory.
        Oldest entries are evicted automatically once the limit is reached.
        Default 100 000.
    """

    def __init__(
        self,
        base_model: str,
        finetuned_model: str,
        rollout_pct: float = 10.0,
        experiment_id: str = "default",
        max_results: int = 100_000,
    ) -> None:
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self._rollout_pct = rollout_pct
        self._experiment_id = experiment_id
        self._results: deque[ABTestResult] = deque(maxlen=max_results)
        self._lock = threading.Lock()

    # ── Routing ───────────────────────────────────────────────────────────────

    @property
    def rollout_pct(self) -> float:
        return self._rollout_pct

    @rollout_pct.setter
    def rollout_pct(self, value: float) -> None:
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"rollout_pct must be in [0, 100], got {value}")
        self._rollout_pct = value

    def route(self, user_id: str) -> tuple[str, Literal["base", "finetuned"]]:
        """
        Return ``(model_identifier, variant_name)`` for *user_id*.

        The assignment is deterministic: the same user always gets the same
        variant while ``rollout_pct`` and ``experiment_id`` remain unchanged.
        """
        if self._hash_bucket(user_id) < self._rollout_pct:
            return self.finetuned_model, "finetuned"
        return self.base_model, "base"

    def record_result(self, result: ABTestResult) -> None:
        """Store an observed A/B result for metric aggregation."""
        with self._lock:
            self._results.append(result)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self, since: datetime | None = None) -> ABTestMetrics:
        """Compute aggregate metrics for both variants.

        Parameters
        ----------
        since:
            Only include results observed after this timestamp.
        """
        with self._lock:
            results = list(self._results)

        if since is not None:
            results = [r for r in results if r.timestamp >= since]

        base = [r for r in results if r.model_variant == "base"]
        ft = [r for r in results if r.model_variant == "finetuned"]

        def _mean(vals: list[float]) -> float:
            return statistics.mean(vals) if vals else 0.0

        def _positive_rate(rs: list[ABTestResult]) -> float:
            scored = [r for r in rs if r.feedback is not None]
            if not scored:
                return 0.0
            return sum(1 for r in scored if r.feedback == "positive") / len(scored)

        def _eval_mean(rs: list[ABTestResult]) -> float | None:
            scores = [r.eval_score for r in rs if r.eval_score is not None]
            return statistics.mean(scores) if scores else None

        return ABTestMetrics(
            base_sample_count=len(base),
            finetuned_sample_count=len(ft),
            base_latency_ms=_mean([r.latency_ms for r in base]),
            finetuned_latency_ms=_mean([r.latency_ms for r in ft]),
            base_cost_usd=_mean([r.cost_usd for r in base]),
            finetuned_cost_usd=_mean([r.cost_usd for r in ft]),
            base_positive_rate=_positive_rate(base),
            finetuned_positive_rate=_positive_rate(ft),
            base_eval_score=_eval_mean(base),
            finetuned_eval_score=_eval_mean(ft),
        )

    def result_count(self) -> int:
        with self._lock:
            return len(self._results)

    def finetuned_count(self) -> int:
        """Return the number of finetuned-variant results currently retained."""
        with self._lock:
            return sum(1 for r in self._results if r.model_variant == "finetuned")

    def clear_results(self) -> None:
        with self._lock:
            self._results.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _hash_bucket(self, user_id: str) -> float:
        """Map *user_id* to a stable float in [0, 100) via SHA-256."""
        digest = hashlib.sha256(f"{self._experiment_id}:{user_id}".encode()).hexdigest()
        # Use first 8 hex chars → 32-bit value
        return int(digest[:8], 16) / 0xFFFFFFFF * 100
