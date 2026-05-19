"""Tests for ContinuousTrainer — orchestrator lifecycle and eval gating."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from synapsekit.training.ab_testing import ABTestRouter
from synapsekit.training.cost_analysis import CostBenefitAnalyzer, InferenceProfile
from synapsekit.training.dataset import TrainingDataGenerator
from synapsekit.training.feedback import FeedbackCollector, InMemoryFeedbackBackend
from synapsekit.training.finetune import AnthropicFineTuneProvider
from synapsekit.training.orchestrator import ContinuousTrainer
from synapsekit.training.rollout import AutoRolloutManager
from synapsekit.training.types import FeedbackSample, FineTuneJob, RolloutPolicy

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_trainer(
    min_feedback: int = 2,
    dataset_path: str | None = None,
    eval_fn: Any = None,
    min_eval_score: float | None = None,
) -> tuple[FeedbackCollector, ContinuousTrainer, AnthropicFineTuneProvider]:
    backend = InMemoryFeedbackBackend()
    collector = FeedbackCollector(backend=backend)
    generator = TrainingDataGenerator()
    provider = AnthropicFineTuneProvider()
    router = ABTestRouter("base-model", "ft-model", rollout_pct=0.0)
    policy = RolloutPolicy(
        stages=[5.0, 25.0, 50.0, 100.0],
        min_samples_per_stage=5,
        improvement_threshold_pct=2.0,
    )
    rollout = AutoRolloutManager(router, policy)
    analyzer = CostBenefitAnalyzer()

    path = dataset_path or tempfile.mktemp(suffix=".jsonl")
    trainer = ContinuousTrainer(
        collector=collector,
        generator=generator,
        provider=provider,
        router=router,
        rollout_manager=rollout,
        analyzer=analyzer,
        base_model="claude-3-haiku-20240307",
        min_feedback_before_train=min_feedback,
        dataset_path=path,
        eval_fn=eval_fn,
        min_eval_score=min_eval_score,
    )
    return collector, trainer, provider


# ── record_feedback ───────────────────────────────────────────────────────────


class TestRecordFeedback:
    def test_record_feedback_is_synchronous(self) -> None:
        _, trainer, _ = _make_trainer()
        sample = trainer.record_feedback("q", "r", "positive")
        assert isinstance(sample, FeedbackSample)

    def test_record_feedback_passes_latency_and_cost(self) -> None:
        _, trainer, _ = _make_trainer()
        sample = trainer.record_feedback("q", "r", "positive", latency_ms=150.0, cost_usd=0.002)
        assert sample.latency_ms == 150.0
        assert sample.cost_usd == 0.002


# ── maybe_trigger_training ────────────────────────────────────────────────────


class TestMaybeTriggerTraining:
    @pytest.mark.asyncio
    async def test_returns_none_below_threshold(self) -> None:
        collector, trainer, _ = _make_trainer(min_feedback=10)
        collector.start()
        collector.record("q", "r", "positive")
        await collector.stop()
        job = await trainer.maybe_trigger_training()
        assert job is None

    @pytest.mark.asyncio
    async def test_returns_job_above_threshold(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            collector, trainer, _ = _make_trainer(min_feedback=2, dataset_path=path)
            collector.start()
            collector.record("q1", "r1", "positive")
            collector.record("q2", "r2", "positive")
            await collector.stop()

            job = await trainer.maybe_trigger_training()
            assert job is not None
            assert isinstance(job, FineTuneJob)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_skips_if_job_already_active(self) -> None:
        collector, trainer, _ = _make_trainer(min_feedback=1)
        collector.start()
        collector.record("q", "r", "positive")
        await collector.stop()

        trainer._pending_job = FineTuneJob(
            job_id="running-id",
            provider="anthropic",
            base_model="claude-3-haiku-20240307",
            status="running",
        )
        job = await trainer.maybe_trigger_training()
        assert job is None

    @pytest.mark.asyncio
    async def test_watermark_only_counts_new_samples(self) -> None:
        """Second trigger call must NOT retrain on samples already used."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            collector, trainer, provider = _make_trainer(min_feedback=2, dataset_path=path)
            collector.start()
            collector.record("q1", "r1", "positive")
            collector.record("q2", "r2", "positive")
            await collector.flush()

            # First trigger — should start a job
            job1 = await trainer.maybe_trigger_training()
            assert job1 is not None
            # Mark job complete so it doesn't block re-trigger
            provider._update_status(job1.job_id, "succeeded", fine_tuned_model="ft-model")
            trainer._pending_job = await provider.status(job1.job_id)

            # Second trigger with no new samples — must NOT start another job
            job2 = await trainer.maybe_trigger_training()
            assert job2 is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_watermark_resets_trigger_on_new_samples(self) -> None:
        """After watermark update, adding min_feedback new samples allows re-trigger."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            collector, trainer, provider = _make_trainer(min_feedback=2, dataset_path=path)
            collector.start()
            collector.record("q1", "r1", "positive")
            collector.record("q2", "r2", "positive")
            await collector.flush()

            job1 = await trainer.maybe_trigger_training()
            assert job1 is not None
            provider._update_status(job1.job_id, "succeeded", fine_tuned_model="ft")
            trainer._pending_job = await provider.status(job1.job_id)

            # Add 2 more new samples
            collector.record("q3", "r3", "positive")
            collector.record("q4", "r4", "positive")
            await collector.stop()

            job2 = await trainer.maybe_trigger_training()
            assert job2 is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── check_job_status ──────────────────────────────────────────────────────────


class TestCheckJobStatus:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_job(self) -> None:
        _, trainer, _ = _make_trainer()
        assert await trainer.check_job_status() is None

    @pytest.mark.asyncio
    async def test_returns_updated_status(self) -> None:
        _, trainer, provider = _make_trainer()
        job = await provider.start_job("file", "claude-3-haiku-20240307")
        trainer._pending_job = job
        provider._update_status(job.job_id, "succeeded", fine_tuned_model="ft")

        updated = await trainer.check_job_status()
        assert updated is not None
        assert updated.status == "succeeded"


# ── activate_ab_test ──────────────────────────────────────────────────────────


class TestActivateABTest:
    @pytest.mark.asyncio
    async def test_activate_without_eval_fn_returns_true(self) -> None:
        _, trainer, _ = _make_trainer()
        result = await trainer.activate_ab_test()
        assert result is True
        assert trainer._rollout_manager.state.is_active is True

    @pytest.mark.asyncio
    async def test_activate_with_passing_eval_fn(self) -> None:
        async def _good_eval(model_id: str) -> float:
            return 0.9

        _, trainer, provider = _make_trainer(eval_fn=_good_eval, min_eval_score=0.7)
        job = await provider.start_job("f", "claude-3-haiku-20240307")
        provider._update_status(job.job_id, "succeeded", fine_tuned_model="ft-model")
        trainer._pending_job = await provider.status(job.job_id)

        result = await trainer.activate_ab_test()
        assert result is True
        assert trainer._rollout_manager.state.is_active is True

    @pytest.mark.asyncio
    async def test_activate_blocked_by_failing_eval(self) -> None:
        async def _bad_eval(model_id: str) -> float:
            return 0.3

        _, trainer, provider = _make_trainer(eval_fn=_bad_eval, min_eval_score=0.7)
        job = await provider.start_job("f", "claude-3-haiku-20240307")
        provider._update_status(job.job_id, "succeeded", fine_tuned_model="ft-model")
        trainer._pending_job = await provider.status(job.job_id)

        result = await trainer.activate_ab_test()
        assert result is False
        assert trainer._rollout_manager.state.is_active is False

    @pytest.mark.asyncio
    async def test_activate_blocked_when_eval_raises(self) -> None:
        async def _crashing_eval(model_id: str) -> float:
            raise RuntimeError("eval service down")

        _, trainer, provider = _make_trainer(eval_fn=_crashing_eval, min_eval_score=0.5)
        job = await provider.start_job("f", "claude-3-haiku-20240307")
        provider._update_status(job.job_id, "succeeded", fine_tuned_model="ft-model")
        trainer._pending_job = await provider.status(job.job_id)

        result = await trainer.activate_ab_test()
        assert result is False

    @pytest.mark.asyncio
    async def test_activate_uses_initial_finetuned_count(self) -> None:
        """activate_ab_test passes current finetuned_count to rollout manager."""
        from synapsekit.training.types import ABTestResult

        _, trainer, _ = _make_trainer()
        router = trainer._router
        # Pre-record some finetuned results before activation
        for i in range(7):
            router.record_result(
                ABTestResult(
                    user_id=f"u{i}",
                    model_variant="finetuned",
                    latency_ms=200.0,
                    cost_usd=0.001,
                )
            )
        await trainer.activate_ab_test()
        assert trainer._rollout_manager.state.stage_start_finetuned_count == 7


# ── run_evaluation_cycle ──────────────────────────────────────────────────────


class TestRunEvaluationCycle:
    @pytest.mark.asyncio
    async def test_returns_metrics(self) -> None:
        _, trainer, _ = _make_trainer()
        metrics, report = await trainer.run_evaluation_cycle()
        assert metrics.base_sample_count == 0
        assert report is None

    @pytest.mark.asyncio
    async def test_returns_cost_report_when_profile_provided(self) -> None:
        _, trainer, _ = _make_trainer()
        profile = InferenceProfile(
            monthly_requests=10_000,
            avg_input_tokens=512,
            avg_output_tokens=256,
            base_model="gpt-4o-mini",
            finetuned_model="gpt-4o-mini:ft",
        )
        trainer.set_training_cost(50.0)
        _, report = await trainer.run_evaluation_cycle(profile=profile)
        assert report is not None
        assert report.training_cost_usd == 50.0


# ── End-to-end pipeline ───────────────────────────────────────────────────────


class TestEndToEndPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        """
        Full cycle: collect → train → activate → record → evaluate → advance.
        """
        from synapsekit.training.types import ABTestResult

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            collector, trainer, provider = _make_trainer(min_feedback=3, dataset_path=path)
            collector.start()

            # 1. Collect feedback
            for i in range(5):
                trainer.record_feedback(f"q{i}", f"r{i}", "positive")
            await collector.flush()

            # 2. Trigger training
            job = await trainer.maybe_trigger_training()
            assert job is not None

            # 3. Simulate job completion
            provider._update_status(job.job_id, "succeeded", fine_tuned_model="ft-model")
            trainer._pending_job = await provider.status(job.job_id)
            trainer.set_training_cost(10.0)

            # 4. Activate A/B test
            activated = await trainer.activate_ab_test()
            assert activated is True

            # 5. Record enough A/B observations to satisfy the sample gate
            router = trainer._router
            for i in range(20):
                router.record_result(
                    ABTestResult(
                        user_id=f"base-{i}",
                        model_variant="base",
                        latency_ms=200.0,
                        cost_usd=0.001,
                        feedback="positive",
                    )
                )
                router.record_result(
                    ABTestResult(
                        user_id=f"ft-{i}",
                        model_variant="finetuned",
                        latency_ms=190.0,
                        cost_usd=0.0009,
                        feedback="positive",
                    )
                )

            # 6. Evaluate and advance
            metrics, _ = await trainer.run_evaluation_cycle()
            # Fine-tuned positive rate == base (all positive) → quality_delta = 0
            # → may be "held" depending on metrics, but pipeline must not raise
            assert metrics is not None

            await collector.stop()
        finally:
            if os.path.exists(path):
                os.unlink(path)
