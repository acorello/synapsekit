"""Tests for CostBenefitAnalyzer — ROI calculations and payback estimates."""

from __future__ import annotations

import math

from synapsekit.training.cost_analysis import (
    PROVIDER_PRICING,
    CostBenefitAnalyzer,
    InferenceProfile,
)
from synapsekit.training.types import ABTestMetrics

# ── Helpers ────────────────────────────────────────────────────────────────────


def _metrics(
    base_pos: float = 0.5,
    ft_pos: float = 0.6,
    base_lat: float = 200.0,
    ft_lat: float = 190.0,
    base_cost: float = 0.01,
    ft_cost: float = 0.008,
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


def _profile(
    monthly_requests: int = 10_000,
    base_model: str = "gpt-4o-mini",
    finetuned_model: str = "gpt-4o-mini:ft",
) -> InferenceProfile:
    return InferenceProfile(
        monthly_requests=monthly_requests,
        avg_input_tokens=512,
        avg_output_tokens=256,
        base_model=base_model,
        finetuned_model=finetuned_model,
    )


# ── Report structure ──────────────────────────────────────────────────────────


class TestReportFields:
    def test_report_has_all_required_fields(self) -> None:
        a = CostBenefitAnalyzer()
        report = a.analyze(50.0, _metrics(), _profile())
        assert hasattr(report, "training_cost_usd")
        assert hasattr(report, "monthly_inference_savings_usd")
        assert hasattr(report, "quality_improvement_pct")
        assert hasattr(report, "estimated_payback_days")

    def test_training_cost_preserved(self) -> None:
        a = CostBenefitAnalyzer()
        report = a.analyze(42.50, _metrics(), _profile())
        assert report.training_cost_usd == 42.50

    def test_quality_improvement_from_metrics(self) -> None:
        a = CostBenefitAnalyzer()
        m = _metrics(base_pos=0.5, ft_pos=0.6)  # 20% improvement
        report = a.analyze(10.0, m, _profile())
        assert abs(report.quality_improvement_pct - 20.0) < 1e-6

    def test_metadata_present(self) -> None:
        a = CostBenefitAnalyzer()
        report = a.analyze(10.0, _metrics(), _profile())
        assert report.metadata is not None
        assert "base_model" in report.metadata
        assert "monthly_requests" in report.metadata


# ── Payback calculation ───────────────────────────────────────────────────────


class TestPayback:
    def test_payback_infinite_when_no_savings(self) -> None:
        """If fine-tuned model costs more, there are no savings → inf payback."""
        a = CostBenefitAnalyzer()
        # Use a profile where finetuned_model has same or higher price as base
        profile = InferenceProfile(
            monthly_requests=10_000,
            avg_input_tokens=512,
            avg_output_tokens=256,
            base_model="gpt-4o-mini",
            finetuned_model="gpt-4o-mini",  # same model → zero savings
        )
        report = a.analyze(100.0, _metrics(), profile)
        # Same model → savings = 0 → payback = inf
        assert math.isinf(report.estimated_payback_days)

    def test_payback_days_calculated_correctly(self) -> None:
        """
        With known monthly savings, verify:
            payback_days = training_cost / (monthly_savings / 30)
        """
        # gpt-4o-mini input=$0.15/M, output=$0.60/M
        # gpt-4o-mini:ft input=$0.30/M, output=$1.20/M
        # gpt-4o-mini cost per request (512 in, 256 out):
        #   = 512/1M * 0.15 + 256/1M * 0.60 = 0.0000768 + 0.0001536 = 0.0002304
        # ft cost per request: = 512/1M * 0.30 + 256/1M * 1.20 = 0.0001536 + 0.0003072 = 0.0004608
        # ft is MORE expensive → savings = monthly_base - monthly_ft < 0 → inf
        # Use a custom cheaper ft model instead
        custom_pricing = {
            "base-model": {"input": 5.0, "output": 15.0},
            "cheap-ft-model": {"input": 2.0, "output": 6.0},
        }
        a = CostBenefitAnalyzer(custom_pricing=custom_pricing)
        profile = InferenceProfile(
            monthly_requests=1_000,
            avg_input_tokens=1_000,
            avg_output_tokens=500,
            base_model="base-model",
            finetuned_model="cheap-ft-model",
        )
        # base cost per request: 1000/1M * 5 + 500/1M * 15 = 0.005 + 0.0075 = 0.0125
        # ft cost per request:   1000/1M * 2 + 500/1M * 6  = 0.002 + 0.003  = 0.005
        # monthly base: 0.0125 * 1000 = 12.5
        # monthly ft:   0.005  * 1000 = 5.0
        # monthly savings: 7.5
        # training_cost: 75 USD
        # payback: 75 / (7.5 / 30) = 75 / 0.25 = 300 days
        report = a.analyze(75.0, _metrics(), profile)
        assert abs(report.monthly_inference_savings_usd - 7.5) < 0.001
        assert abs(report.estimated_payback_days - 300.0) < 0.01

    def test_payback_zero_when_no_training_cost(self) -> None:
        custom = {"base": {"input": 5.0, "output": 15.0}, "ft": {"input": 1.0, "output": 3.0}}
        a = CostBenefitAnalyzer(custom_pricing=custom)
        profile = InferenceProfile(1000, 1000, 500, "base", "ft")
        report = a.analyze(0.0, _metrics(), profile)
        assert report.estimated_payback_days == 0.0


# ── Monthly cost / savings ────────────────────────────────────────────────────


class TestMonthlyCost:
    def test_monthly_savings_positive_for_cheaper_model(self) -> None:
        custom = {"base": {"input": 10.0, "output": 20.0}, "ft": {"input": 1.0, "output": 2.0}}
        a = CostBenefitAnalyzer(custom_pricing=custom)
        profile = InferenceProfile(10_000, 500, 200, "base", "ft")
        report = a.analyze(50.0, _metrics(), profile)
        assert report.monthly_inference_savings_usd > 0

    def test_monthly_savings_negative_for_more_expensive_model(self) -> None:
        custom = {"base": {"input": 1.0, "output": 2.0}, "ft": {"input": 10.0, "output": 20.0}}
        a = CostBenefitAnalyzer(custom_pricing=custom)
        profile = InferenceProfile(10_000, 500, 200, "base", "ft")
        report = a.analyze(50.0, _metrics(), profile)
        assert report.monthly_inference_savings_usd < 0

    def test_no_finetuned_model_falls_back_to_base(self) -> None:
        """When finetuned_model is None, both paths use base_model → zero savings."""
        a = CostBenefitAnalyzer()
        profile = InferenceProfile(10_000, 500, 200, "gpt-4o-mini", finetuned_model=None)
        report = a.analyze(10.0, _metrics(), profile)
        # Same model both sides → savings = 0
        assert report.monthly_inference_savings_usd == 0.0


# ── Training cost estimation ──────────────────────────────────────────────────


class TestEstimateTrainingCost:
    def test_estimate_positive(self) -> None:
        a = CostBenefitAnalyzer()
        cost = a.estimate_training_cost(n_examples=1000, avg_tokens_per_example=500)
        assert cost > 0

    def test_estimate_scales_with_examples(self) -> None:
        a = CostBenefitAnalyzer()
        c1 = a.estimate_training_cost(1000, 500)
        c2 = a.estimate_training_cost(2000, 500)
        assert abs(c2 / c1 - 2.0) < 1e-9

    def test_estimate_scales_with_epochs(self) -> None:
        a = CostBenefitAnalyzer()
        c1 = a.estimate_training_cost(1000, 500, n_epochs=1)
        c3 = a.estimate_training_cost(1000, 500, n_epochs=3)
        assert abs(c3 / c1 - 3.0) < 1e-9


# ── Token cost ────────────────────────────────────────────────────────────────


class TestTokenCost:
    def test_token_cost_gpt4o_mini(self) -> None:
        a = CostBenefitAnalyzer()
        # 1M input + 1M output = 0.15 + 0.60 = 0.75
        cost = a.token_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert abs(cost - 0.75) < 1e-9

    def test_token_cost_unknown_model_is_zero(self) -> None:
        a = CostBenefitAnalyzer()
        cost = a.token_cost("nonexistent-model-xyz", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_token_cost_custom_pricing(self) -> None:
        a = CostBenefitAnalyzer(custom_pricing={"my-model": {"input": 2.0, "output": 4.0}})
        cost = a.token_cost("my-model", 500_000, 500_000)
        # 0.5M * 2 + 0.5M * 4 = 1.0 + 2.0 = 3.0
        assert abs(cost - 3.0) < 1e-9


# ── Pricing table ─────────────────────────────────────────────────────────────


class TestPricingTable:
    def test_provider_pricing_has_gpt4o_mini(self) -> None:
        assert "gpt-4o-mini" in PROVIDER_PRICING

    def test_provider_pricing_has_input_output_keys(self) -> None:
        for model, rates in PROVIDER_PRICING.items():
            assert "input" in rates, f"{model} missing 'input' key"
            assert "output" in rates, f"{model} missing 'output' key"
