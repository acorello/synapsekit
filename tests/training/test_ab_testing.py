"""Tests for ABTestRouter — routing, sticky assignment, and metric aggregation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synapsekit.training.ab_testing import ABTestRouter
from synapsekit.training.types import ABTestMetrics, ABTestResult

# ── Helpers ────────────────────────────────────────────────────────────────────


def _router(pct: float = 50.0) -> ABTestRouter:
    return ABTestRouter("base-v1", "ft-v1", rollout_pct=pct, experiment_id="test-exp")


def _result(
    user_id: str = "u1",
    variant: str = "base",
    latency: float = 200.0,
    cost: float = 0.001,
    feedback: str | None = "positive",
    score: float | None = None,
) -> ABTestResult:
    return ABTestResult(
        user_id=user_id,
        model_variant=variant,  # type: ignore[arg-type]
        latency_ms=latency,
        cost_usd=cost,
        feedback=feedback,  # type: ignore[arg-type]
        eval_score=score,
    )


# ── Routing ────────────────────────────────────────────────────────────────────


class TestRouting:
    def test_route_returns_tuple_of_model_and_variant(self) -> None:
        r = _router(50.0)
        model, variant = r.route("user-abc")
        assert model in ("base-v1", "ft-v1")
        assert variant in ("base", "finetuned")

    def test_route_deterministic_for_same_user(self) -> None:
        r = _router(50.0)
        first = r.route("user-abc")
        second = r.route("user-abc")
        assert first == second

    def test_route_0_pct_always_base(self) -> None:
        r = _router(0.0)
        for uid in [f"user-{i}" for i in range(50)]:
            _, variant = r.route(uid)
            assert variant == "base"

    def test_route_100_pct_always_finetuned(self) -> None:
        r = _router(100.0)
        for uid in [f"user-{i}" for i in range(50)]:
            _, variant = r.route(uid)
            assert variant == "finetuned"

    def test_route_distributes_traffic(self) -> None:
        """With 50% rollout, roughly half of users land on each variant."""
        r = _router(50.0)
        variants = [r.route(f"user-{i}")[1] for i in range(200)]
        ft_count = sum(1 for v in variants if v == "finetuned")
        assert 60 <= ft_count <= 140

    def test_different_experiment_ids_give_different_assignments(self) -> None:
        r1 = ABTestRouter("base", "ft", rollout_pct=50.0, experiment_id="exp-A")
        r2 = ABTestRouter("base", "ft", rollout_pct=50.0, experiment_id="exp-B")
        variants_1 = {f"u{i}": r1.route(f"u{i}")[1] for i in range(20)}
        variants_2 = {f"u{i}": r2.route(f"u{i}")[1] for i in range(20)}
        diff = sum(1 for k in variants_1 if variants_1[k] != variants_2[k])
        assert diff > 0


# ── Rollout pct setter ────────────────────────────────────────────────────────


class TestRolloutPctSetter:
    def test_set_valid_pct(self) -> None:
        r = _router()
        r.rollout_pct = 25.0
        assert r.rollout_pct == 25.0

    def test_set_0_valid(self) -> None:
        r = _router()
        r.rollout_pct = 0.0
        assert r.rollout_pct == 0.0

    def test_set_100_valid(self) -> None:
        r = _router()
        r.rollout_pct = 100.0
        assert r.rollout_pct == 100.0

    def test_set_negative_raises(self) -> None:
        r = _router()
        with pytest.raises(ValueError):
            r.rollout_pct = -1.0

    def test_set_above_100_raises(self) -> None:
        r = _router()
        with pytest.raises(ValueError):
            r.rollout_pct = 101.0


# ── Metric aggregation ────────────────────────────────────────────────────────


class TestMetrics:
    def test_empty_metrics_returns_zeros(self) -> None:
        r = _router()
        m = r.get_metrics()
        assert m.base_sample_count == 0
        assert m.finetuned_sample_count == 0
        assert m.base_latency_ms == 0.0
        assert m.base_positive_rate == 0.0

    def test_positive_rate_computed_correctly(self) -> None:
        r = _router()
        r.record_result(_result("u1", "finetuned", feedback="positive"))
        r.record_result(_result("u2", "finetuned", feedback="positive"))
        r.record_result(_result("u3", "finetuned", feedback="negative"))
        m = r.get_metrics()
        assert abs(m.finetuned_positive_rate - 2 / 3) < 1e-9

    def test_latency_mean_computed(self) -> None:
        r = _router()
        r.record_result(_result("u1", "base", latency=100.0))
        r.record_result(_result("u2", "base", latency=200.0))
        m = r.get_metrics()
        assert m.base_latency_ms == 150.0

    def test_eval_score_mean(self) -> None:
        r = _router()
        r.record_result(_result("u1", "finetuned", score=0.8))
        r.record_result(_result("u2", "finetuned", score=0.6))
        m = r.get_metrics()
        assert abs(m.finetuned_eval_score - 0.7) < 1e-9

    def test_eval_score_none_when_no_scores(self) -> None:
        r = _router()
        r.record_result(_result("u1", "base", score=None))
        m = r.get_metrics()
        assert m.base_eval_score is None

    def test_result_count(self) -> None:
        r = _router()
        r.record_result(_result("u1", "base"))
        r.record_result(_result("u2", "finetuned"))
        assert r.result_count() == 2

    def test_clear_results(self) -> None:
        r = _router()
        r.record_result(_result())
        r.clear_results()
        assert r.result_count() == 0

    def test_since_filter(self) -> None:
        r = _router()
        old = ABTestResult(
            user_id="u1",
            model_variant="base",
            latency_ms=100.0,
            cost_usd=0.001,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        r.record_result(old)
        r.record_result(_result("u2", "base"))
        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        m = r.get_metrics(since=cutoff)
        assert m.base_sample_count == 1


# ── Bounded results (max_results) ─────────────────────────────────────────────


class TestBoundedResults:
    def test_max_results_caps_memory(self) -> None:
        """_results deque must not grow beyond max_results."""
        r = ABTestRouter("base", "ft", rollout_pct=50.0, max_results=10)
        for i in range(25):
            r.record_result(_result(f"u{i}", "base"))
        assert r.result_count() == 10

    def test_oldest_results_evicted(self) -> None:
        """When max_results is exceeded the oldest entries are dropped."""
        r = ABTestRouter("base", "ft", rollout_pct=50.0, max_results=5)
        # Record 5 base + 5 finetuned; deque holds only last 5
        for i in range(5):
            r.record_result(_result(f"b{i}", "base"))
        for i in range(5):
            r.record_result(_result(f"f{i}", "finetuned"))
        # All 5 retained are finetuned (the 5 base were evicted)
        m = r.get_metrics()
        assert m.base_sample_count == 0
        assert m.finetuned_sample_count == 5

    def test_finetuned_count_helper(self) -> None:
        r = _router(50.0)
        r.record_result(_result("u1", "base"))
        r.record_result(_result("u2", "finetuned"))
        r.record_result(_result("u3", "finetuned"))
        assert r.finetuned_count() == 2


# ── ABTestMetrics computed properties ─────────────────────────────────────────


class TestABTestMetricsProperties:
    def _m(
        self,
        base_lat: float = 200.0,
        ft_lat: float = 200.0,
        base_pos: float = 0.5,
        ft_pos: float = 0.6,
        base_cost: float = 0.01,
        ft_cost: float = 0.01,
    ) -> ABTestMetrics:
        return ABTestMetrics(
            base_sample_count=100,
            finetuned_sample_count=100,
            base_latency_ms=base_lat,
            finetuned_latency_ms=ft_lat,
            base_cost_usd=base_cost,
            finetuned_cost_usd=ft_cost,
            base_positive_rate=base_pos,
            finetuned_positive_rate=ft_pos,
        )

    def test_latency_delta_zero_when_equal(self) -> None:
        m = self._m(base_lat=200.0, ft_lat=200.0)
        assert m.latency_delta_pct == 0.0

    def test_latency_delta_positive_when_slower(self) -> None:
        m = self._m(base_lat=200.0, ft_lat=240.0)
        assert abs(m.latency_delta_pct - 20.0) < 1e-9

    def test_quality_delta_positive_when_improved(self) -> None:
        m = self._m(base_pos=0.5, ft_pos=0.6)
        assert abs(m.quality_delta_pct - 20.0) < 1e-9

    def test_quality_delta_negative_when_worse(self) -> None:
        m = self._m(base_pos=0.6, ft_pos=0.5)
        assert m.quality_delta_pct < 0

    def test_cost_delta_zero_when_equal(self) -> None:
        m = self._m(base_cost=0.01, ft_cost=0.01)
        assert m.cost_delta_pct == 0.0

    def test_latency_delta_zero_when_base_is_zero(self) -> None:
        m = self._m(base_lat=0.0, ft_lat=100.0)
        assert m.latency_delta_pct == 0.0
