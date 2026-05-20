"""Continuous fine-tuning orchestrator — ContinuousTrainer."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from .ab_testing import ABTestRouter
from .cost_analysis import CostBenefitAnalyzer, InferenceProfile
from .dataset import TrainingDataGenerator
from .feedback import FeedbackCollector
from .finetune import BaseFineTuneProvider
from .rollout import AutoRolloutManager
from .types import CostBenefitReport, FeedbackSample, FineTuneJob

_log = logging.getLogger(__name__)

# Public re-export so callers can do: from synapsekit.training.orchestrator import ABTestMetrics
from .types import ABTestMetrics  # noqa: E402  (re-export)


class ContinuousTrainer:
    """
    High-level orchestrator for the closed-loop continuous improvement cycle.

    Coordinates:
        feedback collection → dataset generation → fine-tuning →
        evaluation → A/B testing → rollout

    Retraining is batch-triggered when *new* accumulated feedback (since the
    last training run) reaches ``min_feedback_before_train``, not on every
    individual sample.

    Parameters
    ----------
    collector:
        FeedbackCollector instance (must be started before use).
    generator:
        TrainingDataGenerator for producing JSONL datasets.
    provider:
        Fine-tuning provider (OpenAI, Anthropic, …).
    router:
        ABTestRouter for traffic splitting.
    rollout_manager:
        AutoRolloutManager controlling gradual rollout.
    analyzer:
        CostBenefitAnalyzer for ROI computation.
    base_model:
        Base model identifier used for fine-tuning jobs.
    min_feedback_before_train:
        Minimum *new* feedback samples (since the last training job was
        started) required to trigger a new training job.  Default 100.
    dataset_path:
        Local path where JSONL datasets are written before upload.
    inference_profile:
        Default production traffic profile for cost analysis.
    eval_fn:
        Optional async callable ``(model_id: str) -> float`` that returns
        an evaluation score for a fine-tuned model.  If provided,
        ``activate_ab_test()`` calls this before starting any rollout.
    min_eval_score:
        If ``eval_fn`` is set and the returned score falls below this
        threshold, rollout activation is blocked.
    """

    def __init__(
        self,
        collector: FeedbackCollector,
        generator: TrainingDataGenerator,
        provider: BaseFineTuneProvider,
        router: ABTestRouter,
        rollout_manager: AutoRolloutManager,
        analyzer: CostBenefitAnalyzer,
        base_model: str,
        min_feedback_before_train: int = 100,
        dataset_path: str = "synapsekit_training.jsonl",
        inference_profile: InferenceProfile | None = None,
        eval_fn: Callable[[str], Awaitable[float]] | None = None,
        min_eval_score: float | None = None,
    ) -> None:
        self._collector = collector
        self._generator = generator
        self._provider = provider
        self._router = router
        self._rollout_manager = rollout_manager
        self._analyzer = analyzer
        self._base_model = base_model
        self._min_feedback = min_feedback_before_train
        self._dataset_path = dataset_path
        self._inference_profile = inference_profile
        self._eval_fn = eval_fn
        self._min_eval_score = min_eval_score
        self._pending_job: FineTuneJob | None = None
        self._training_cost_usd: float = 0.0
        # Watermark: count of stored samples included in the last training run.
        # Only NEW samples (stored_count - _trained_up_to) count toward the
        # min_feedback_before_train threshold, preventing unbounded re-training
        # of ever-growing historical datasets.
        self._trained_up_to: int = 0

    # ── Feedback ──────────────────────────────────────────────────────────────

    def record_feedback(
        self,
        query: str,
        response: str,
        feedback: Literal["positive", "negative"],
        **kwargs: Any,
    ) -> FeedbackSample:
        """
        Record a production feedback sample.

        Delegates to FeedbackCollector.record() — synchronous and non-blocking.
        Accepts all FeedbackSample keyword arguments including latency_ms and
        cost_usd.
        """
        return self._collector.record(query, response, feedback, **kwargs)

    # ── Training lifecycle ────────────────────────────────────────────────────

    async def maybe_trigger_training(self) -> FineTuneJob | None:
        """
        Check whether enough *new* feedback has accumulated to start a job.

        Skips silently if a job is already active (queued / running).
        Returns the new FineTuneJob if training was triggered, else None.
        """
        if self._pending_job and self._pending_job.status in ("queued", "running"):
            return None

        current_count = await self._collector.stored_count()
        new_samples = current_count - self._trained_up_to
        if new_samples < self._min_feedback:
            return None

        samples = await self._collector.get_samples()
        # Write JSONL in a thread to avoid blocking the event loop.
        n_written = await asyncio.to_thread(
            self._generator.write_jsonl, samples, self._dataset_path
        )
        if n_written == 0:
            _log.info("No training examples generated — skipping job.")
            return None

        _log.info("Starting fine-tune job: %d examples → %s", n_written, self._dataset_path)
        file_id = await self._provider.upload_dataset(self._dataset_path)
        job = await self._provider.start_job(file_id, self._base_model)
        self._pending_job = job
        self._trained_up_to = current_count
        return job

    async def check_job_status(self) -> FineTuneJob | None:
        """Poll and return the current status of the pending fine-tune job."""
        if self._pending_job is None:
            return None
        self._pending_job = await self._provider.status(self._pending_job.job_id)
        return self._pending_job

    # ── Rollout ───────────────────────────────────────────────────────────────

    async def activate_ab_test(self, model_id: str | None = None) -> bool:
        """
        Activate the rollout, optionally gating on an evaluation score.

        If ``eval_fn`` was supplied at construction, it is called with the
        fine-tuned model ID before any rollout traffic is sent.  If the score
        falls below ``min_eval_score``, activation is blocked and False is
        returned.

        Parameters
        ----------
        model_id:
            Override the fine-tuned model ID to evaluate.  Falls back to
            the model ID stored on the pending job.

        Returns
        -------
        True if the rollout was activated, False if the eval gate blocked it.
        """
        if self._eval_fn is not None:
            effective_id = model_id or (
                self._pending_job.fine_tuned_model if self._pending_job else None
            )
            if effective_id is not None:
                try:
                    score = await self._eval_fn(effective_id)
                except Exception:
                    _log.exception("Evaluation failed for model %s — rollout blocked", effective_id)
                    return False
                if self._min_eval_score is not None and score < self._min_eval_score:
                    _log.warning(
                        "Eval score %.3f below threshold %.3f for %s — rollout blocked",
                        score,
                        self._min_eval_score,
                        effective_id,
                    )
                    return False

        initial_ft_count = self._router.finetuned_count()
        self._rollout_manager.activate(initial_finetuned_count=initial_ft_count)
        return True

    async def run_evaluation_cycle(
        self,
        profile: InferenceProfile | None = None,
        training_cost_usd: float | None = None,
    ) -> tuple[ABTestMetrics, CostBenefitReport | None]:
        """
        Evaluate current A/B metrics and optionally compute cost-benefit.

        Parameters
        ----------
        profile:
            InferenceProfile for cost projection; falls back to the instance
            default set during construction.
        training_cost_usd:
            Override the stored training cost for this analysis.

        Returns
        -------
        (ABTestMetrics, CostBenefitReport | None)
        """
        metrics = self._router.get_metrics()
        self._rollout_manager.evaluate_and_advance(metrics)

        cost = training_cost_usd if training_cost_usd is not None else self._training_cost_usd
        eff_profile = profile or self._inference_profile
        report: CostBenefitReport | None = None
        if eff_profile is not None:
            report = self._analyzer.analyze(cost, metrics, eff_profile)

        return metrics, report

    def set_training_cost(self, cost_usd: float) -> None:
        """Store the training cost for subsequent cost-benefit analysis."""
        self._training_cost_usd = cost_usd

    @property
    def pending_job(self) -> FineTuneJob | None:
        return self._pending_job
