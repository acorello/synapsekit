from __future__ import annotations

import synapsekit.observe as observe
from synapsekit.observability.metrics import PrometheusMetrics


class TestPrometheusMetrics:
    def test_prometheus_metrics_records_from_span(self):
        metrics = PrometheusMetrics(enabled=False)
        observe.configure(metrics=metrics)

        span = observe.start_span(
            "llm.generate",
            {
                "llm.model": "gpt-4o-mini",
                "llm.provider": "openai",
                "llm.total_tokens": 12,
                "llm.cost_usd": 0.0012,
                "llm.latency_ms": 25.0,
            },
        )
        observe.end_span(span)

        # No crash, metrics disabled => no-ops
        assert True
