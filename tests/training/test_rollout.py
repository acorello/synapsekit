"""Tests for AutoRolloutManager — advancement, holds, and rollback conditions."""

from __future__ import annotations

from synapsekit.training.ab_testing import ABTestRouter
from synapsekit.training.rollout import AutoRolloutManager
from synapsekit.training.types import ABTestMetrics, RolloutPolicy

# ── Helpers ────────────────────────────────────────────────────────────────────


def _router() -> ABTestRouter:
    return ABTestRouter("base", "ft", rollout_pct=0.0)


def _manager(
    stages: list[float] | None = None,
    min_samples: int = 10,
    improvement_pct: float = 2.0,
    latency_reg: float = 20.0,
    cost_reg: float = 15.0,
    router: ABTestRouter | None = None,
) -> AutoRolloutManager:
    policy = RolloutPolicy(
        stages=stages or [5.0, 25.0, 50.0, 100.0],
        min_samples_per_stage=min_samples,
        improvement_threshold_pct=improvement_pct,
        latency_regression_pct=latency_reg,
        cost_regression_pct=cost_reg,
    )
    r = router or _router()
    return AutoRolloutManager(router=r, policy=policy)


def _metrics(
    ft_samples: int = 50,
    quality_delta: float = 5.0,
    latency_delta: float = 0.0,
    cost_delta: float = 0.0,
    base_pos: float = 0.5,
    ft_offset: int = 0,
) -> ABTestMetrics:
    """
    Build ABTestMetrics with specific delta percentages.

    ft_offset is subtracted from ft_samples to simulate the
    stage_start_finetuned_count baseline so that
    ``ft_samples - ft_offset`` new samples are seen since stage start.
    """
    ft_pos = base_pos * (1 + quality_delta / 100)
    base_lat = 200.0
    ft_lat = base_lat * (1 + latency_delta / 100)
    base_cost = 0.01
    ft_cost = base_cost * (1 + cost_delta / 100)
    return ABTestMetrics(
        base_sample_count=100,
        finetuned_sample_count=ft_samples,
        base_latency_ms=base_lat,
        finetuned_latency_ms=ft_lat,
        base_cost_usd=base_cost,
        finetuned_cost_usd=ft_cost,
        base_positive_rate=base_pos,
        finetuned_positive_rate=ft_pos,
    )


# ── Activation ────────────────────────────────────────────────────────────────


class TestActivation:
    def test_activate_sets_first_stage_pct(self) -> None:
        m = _manager(stages=[5.0, 25.0, 50.0, 100.0])
        m.activate()
        assert m.state.current_pct == 5.0

    def test_activate_sets_router_pct(self) -> None:
        r = _router()
        m = _manager(router=r)
        m.activate()
        assert r.rollout_pct == 5.0

    def test_activate_sets_is_active(self) -> None:
        m = _manager()
        m.activate()
        assert m.state.is_active is True

    def test_activate_clears_rollback_state(self) -> None:
        m = _manager()
        m.rollback("old reason")
        m.activate()
        assert m.state.rolled_back is False
        assert m.state.rollback_reason is None

    def test_activate_sets_stage_start_finetuned_count(self) -> None:
        m = _manager()
        m.activate(initial_finetuned_count=42)
        assert m.state.stage_start_finetuned_count == 42

    def test_activate_default_initial_count_zero(self) -> None:
        m = _manager()
        m.activate()
        assert m.state.stage_start_finetuned_count == 0


# ── Advancement ───────────────────────────────────────────────────────────────


class TestAdvancement:
    def test_advance_on_sufficient_samples_and_improvement(self) -> None:
        m = _manager(min_samples=10, improvement_pct=2.0)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert result == "advanced"

    def test_advance_increases_rollout_pct(self) -> None:
        r = _router()
        m = _manager(stages=[5.0, 25.0], min_samples=10, router=r)
        m.activate()
        m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert r.rollout_pct == 25.0

    def test_hold_insufficient_samples(self) -> None:
        m = _manager(min_samples=100)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert result == "held"

    def test_hold_below_improvement_threshold(self) -> None:
        m = _manager(improvement_pct=5.0)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=1.0))
        assert result == "held"

    def test_hold_when_not_active(self) -> None:
        m = _manager()
        result = m.evaluate_and_advance(_metrics())
        assert result == "held"

    def test_completed_at_last_stage(self) -> None:
        m = _manager(stages=[5.0], min_samples=10)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert result == "completed"

    def test_completed_sets_100_pct(self) -> None:
        r = _router()
        m = _manager(stages=[5.0], min_samples=10, router=r)
        m.activate()
        m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert r.rollout_pct == 100.0

    def test_completed_sets_is_active_false(self) -> None:
        """After 100% rollout, is_active must be False so stale rollback checks don't fire."""
        m = _manager(stages=[5.0], min_samples=10)
        m.activate()
        m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert m.state.is_active is False

    def test_evaluate_after_completion_returns_held(self) -> None:
        """evaluate_and_advance on a completed manager must return 'held', not 'rolled_back'."""
        m = _manager(stages=[5.0], min_samples=10)
        m.activate()
        m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        # Call again — should be held, NOT rolled_back on noisy metrics
        result = m.evaluate_and_advance(
            _metrics(ft_samples=200, quality_delta=-10.0, latency_delta=50.0)
        )
        assert result == "held"
        assert not m.state.rolled_back

    def test_sequential_advancement_through_stages(self) -> None:
        """Each stage requires NEW samples collected since that stage started."""
        r = _router()
        m = _manager(stages=[5.0, 25.0, 50.0], min_samples=10, router=r)
        m.activate()

        # Stage 0 → 1: 50 new samples (baseline 0)
        r1 = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert r1 == "advanced"
        assert r.rollout_pct == 25.0
        assert m.state.stage_start_finetuned_count == 50

        # Stage 1 → 2: 50 new samples since baseline 50 (total 100)
        r2 = m.evaluate_and_advance(_metrics(ft_samples=100, quality_delta=5.0))
        assert r2 == "advanced"
        assert r.rollout_pct == 50.0
        assert m.state.stage_start_finetuned_count == 100

        # Stage 2 → completed: 50 new samples since baseline 100 (total 150)
        r3 = m.evaluate_and_advance(_metrics(ft_samples=150, quality_delta=5.0))
        assert r3 == "completed"
        assert r.rollout_pct == 100.0

    def test_per_stage_sample_gate_resets_on_advance(self) -> None:
        """After advancing, same cumulative count must not satisfy next stage gate."""
        m = _manager(min_samples=10)
        m.activate()
        # First advance with 50 total finetuned samples
        m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert m.state.stage_start_finetuned_count == 50
        # Calling again with the same 50 total → 0 new samples → held
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert result == "held"


# ── Rollback conditions ────────────────────────────────────────────────────────


class TestRollback:
    def test_rollback_on_latency_regression(self) -> None:
        m = _manager(latency_reg=20.0)
        m.activate()
        result = m.evaluate_and_advance(
            _metrics(ft_samples=50, quality_delta=5.0, latency_delta=25.0)
        )
        assert result == "rolled_back"
        assert m.state.rolled_back is True
        assert "latency" in (m.state.rollback_reason or "").lower()

    def test_rollback_on_cost_regression(self) -> None:
        m = _manager(cost_reg=15.0)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0, cost_delta=20.0))
        assert result == "rolled_back"
        assert "cost" in (m.state.rollback_reason or "").lower()

    def test_rollback_on_quality_regression(self) -> None:
        m = _manager(improvement_pct=2.0, min_samples=10)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=-5.0))
        assert result == "rolled_back"
        assert "quality" in (m.state.rollback_reason or "").lower()

    def test_rollback_resets_router_to_0(self) -> None:
        r = _router()
        m = _manager(latency_reg=20.0, router=r)
        m.activate()
        m.evaluate_and_advance(_metrics(ft_samples=50, latency_delta=30.0))
        assert r.rollout_pct == 0.0

    def test_manual_rollback(self) -> None:
        r = _router()
        m = _manager(router=r)
        m.activate()
        m.rollback("manual test")
        assert r.rollout_pct == 0.0
        assert m.state.rolled_back is True
        assert m.state.rollback_reason == "manual test"

    def test_manual_rollback_disables_active(self) -> None:
        m = _manager()
        m.activate()
        m.rollback()
        assert m.state.is_active is False

    def test_no_rollback_for_quality_below_threshold_count(self) -> None:
        """Quality drop should only trigger rollback when stage sample count is sufficient."""
        m = _manager(improvement_pct=2.0, min_samples=100)
        m.activate()
        # Only 5 new samples — below min_samples, so quality drop is ignored
        result = m.evaluate_and_advance(_metrics(ft_samples=5, quality_delta=-10.0))
        assert result == "held"
        assert not m.state.rolled_back

    def test_no_rollback_on_latency_regression_insufficient_samples(self) -> None:
        """Latency regression must not rollback when stage samples < min_samples_per_stage."""
        m = _manager(latency_reg=20.0, min_samples=50)
        m.activate()
        # Only 5 new samples — not enough to act on latency noise
        result = m.evaluate_and_advance(_metrics(ft_samples=5, latency_delta=99.0))
        assert result == "held"
        assert not m.state.rolled_back

    def test_no_rollback_on_cost_regression_insufficient_samples(self) -> None:
        """Cost regression must not rollback when stage samples < min_samples_per_stage."""
        m = _manager(cost_reg=15.0, min_samples=50)
        m.activate()
        result = m.evaluate_and_advance(_metrics(ft_samples=3, cost_delta=99.0))
        assert result == "held"
        assert not m.state.rolled_back

    def test_held_after_rollback(self) -> None:
        """evaluate_and_advance on a rolled-back manager returns held."""
        m = _manager()
        m.activate()
        m.rollback("manual")
        result = m.evaluate_and_advance(_metrics(ft_samples=50, quality_delta=5.0))
        assert result == "held"
