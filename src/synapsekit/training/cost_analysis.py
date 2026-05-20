"""Cost-benefit analysis for fine-tuning investments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from .types import ABTestMetrics, CostBenefitReport

# Per-model pricing in USD per 1 million tokens (input / output).
# Update as providers revise their rate cards.
PROVIDER_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 3.0, "output": 6.0},
    # Fine-tuned variants share the same base price key;
    # callers can override via custom_pricing.
    "gpt-4o-mini:ft": {"input": 0.30, "output": 1.20},
    "gpt-3.5-turbo:ft": {"input": 3.0, "output": 6.0},
    # Anthropic
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
}

# Training cost per 1M tokens (OpenAI fine-tuning, as of 2025).
OPENAI_TRAINING_RATE_PER_M_TOKENS: dict[str, float] = {
    "gpt-4o-mini": 8.0,
    "gpt-3.5-turbo": 8.0,
    "gpt-4o": 25.0,
}

# Training cost per 1M tokens (Anthropic fine-tuning — estimated enterprise rates).
ANTHROPIC_TRAINING_RATE_PER_M_TOKENS: dict[str, float] = {
    "claude-3-haiku-20240307": 2.0,
    "claude-3-5-haiku-20241022": 2.0,
    "claude-3-5-sonnet-20241022": 15.0,
}

_TRAINING_RATES: dict[str, dict[str, float]] = {
    "openai": OPENAI_TRAINING_RATE_PER_M_TOKENS,
    "anthropic": ANTHROPIC_TRAINING_RATE_PER_M_TOKENS,
}


@dataclass
class InferenceProfile:
    """Describes production traffic for monthly cost projection."""

    monthly_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    base_model: str
    finetuned_model: str | None = None


class CostBenefitAnalyzer:
    """
    Computes training cost, inference savings, quality gains, and ROI
    for a completed or projected fine-tuning cycle.

    All monetary values are in USD.

    Parameters
    ----------
    custom_pricing:
        Override or extend the default ``PROVIDER_PRICING`` table.
        Keyed by model identifier; each value is ``{"input": ..., "output": ...}``
        in USD per 1M tokens.
    """

    def __init__(self, custom_pricing: dict[str, dict[str, float]] | None = None) -> None:
        self._pricing = {**PROVIDER_PRICING, **(custom_pricing or {})}

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        training_cost_usd: float,
        metrics: ABTestMetrics,
        profile: InferenceProfile,
    ) -> CostBenefitReport:
        """
        Compute the full cost-benefit report.

        Parameters
        ----------
        training_cost_usd:
            Actual or estimated cost of the fine-tuning job.
        metrics:
            A/B test metrics comparing base vs fine-tuned variants.
        profile:
            Production traffic profile for monthly cost projection.
        """
        monthly_base = self._monthly_cost(profile, use_finetuned=False)
        monthly_ft = self._monthly_cost(profile, use_finetuned=True)
        monthly_savings = monthly_base - monthly_ft

        if monthly_savings <= 0 or training_cost_usd == 0:
            payback_days = float("inf") if monthly_savings <= 0 else 0.0
        else:
            daily_savings = monthly_savings / 30.0
            payback_days = training_cost_usd / daily_savings

        return CostBenefitReport(
            training_cost_usd=training_cost_usd,
            monthly_inference_savings_usd=monthly_savings,
            quality_improvement_pct=metrics.quality_delta_pct,
            estimated_payback_days=payback_days,
            metadata={
                "base_model": profile.base_model,
                "finetuned_model": profile.finetuned_model,
                "monthly_requests": profile.monthly_requests,
                "avg_input_tokens": profile.avg_input_tokens,
                "avg_output_tokens": profile.avg_output_tokens,
                "monthly_base_cost_usd": monthly_base,
                "monthly_ft_cost_usd": monthly_ft,
                "base_latency_ms": metrics.base_latency_ms,
                "finetuned_latency_ms": metrics.finetuned_latency_ms,
                "latency_delta_pct": metrics.latency_delta_pct,
                "cost_delta_pct": metrics.cost_delta_pct,
            },
        )

    def estimate_training_cost(
        self,
        n_examples: int,
        avg_tokens_per_example: int,
        n_epochs: int = 3,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> float:
        """
        Estimate fine-tuning training cost in USD.

        Parameters
        ----------
        provider:
            "openai" or "anthropic".  Selects the correct rate table.
        """
        rate_table = _TRAINING_RATES.get(provider)
        if rate_table is None:
            warnings.warn(
                f"Unknown provider {provider!r} for training cost estimation; "
                "falling back to OpenAI gpt-4o-mini rate (8.0 USD/M tokens).",
                stacklevel=2,
            )
            rate = 8.0
        else:
            _looked_up = rate_table.get(model)
            if _looked_up is None:
                warnings.warn(
                    f"Model {model!r} not found in {provider!r} training rate table; "
                    "using default rate of 8.0 USD/M tokens.",
                    stacklevel=2,
                )
                rate = 8.0
            else:
                rate = _looked_up

        total_tokens = n_examples * avg_tokens_per_example * n_epochs
        return total_tokens / 1_000_000 * rate

    def token_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return the USD cost of a single inference call for *model*."""
        pricing = self._pricing.get(model)
        if pricing is None:
            warnings.warn(
                f"Model {model!r} not found in pricing table; cost returned as 0.0. "
                "Pass custom_pricing to CostBenefitAnalyzer to suppress this warning.",
                stacklevel=2,
            )
            return 0.0
        return (
            input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"]
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _monthly_cost(self, profile: InferenceProfile, use_finetuned: bool) -> float:
        model = (
            profile.finetuned_model
            if (use_finetuned and profile.finetuned_model)
            else profile.base_model
        )
        per_request = self.token_cost(model, profile.avg_input_tokens, profile.avg_output_tokens)
        return per_request * profile.monthly_requests
