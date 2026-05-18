"""Prometheus metrics exporter for SynapseKit observability."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

try:  # pragma: no cover - avoid circular import at runtime
    from ..observe.spans import SpanAttributes
except Exception:  # pragma: no cover
    SpanAttributes = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Counter = Histogram = CollectorRegistry = start_http_server = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False


class PrometheusMetrics:
    """Prometheus metrics for LLM cost, tokens, and latency.

    Metrics:
      - synapsekit_cost_usd_total
      - synapsekit_tokens_total
      - synapsekit_latency_seconds (histogram)
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        namespace: str = "synapsekit",
        registry: Any | None = None,
        start_server: bool = False,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.enabled = bool(enabled) and _PROMETHEUS_AVAILABLE
        self._namespace = namespace
        self._registry = registry or (CollectorRegistry() if self.enabled else None)
        self._server_started = False

        if self.enabled:
            self._cost_counter = Counter(
                "cost_usd_total",
                "Total LLM cost in USD.",
                ["model", "provider"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._token_counter = Counter(
                "tokens_total",
                "Total LLM tokens.",
                ["model", "provider"],
                namespace=self._namespace,
                registry=self._registry,
            )
            self._latency_hist = Histogram(
                "latency_seconds",
                "LLM latency in seconds.",
                ["model", "provider"],
                namespace=self._namespace,
                registry=self._registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            )
        else:
            self._cost_counter = None
            self._token_counter = None
            self._latency_hist = None

        if start_server and self.enabled:
            self.start_http_server(host=host, port=port)

    def start_http_server(self, *, host: str = "0.0.0.0", port: int = 8000) -> None:
        if not self.enabled or self._server_started:
            return
        if start_http_server is None:
            return
        start_http_server(port, addr=host, registry=self._registry)
        self._server_started = True

    def record_llm(
        self,
        *,
        model: str,
        provider: str,
        cost_usd: float | None,
        total_tokens: int | None,
        latency_ms: float | None,
    ) -> None:
        if not self.enabled:
            return
        if model is None:
            model = "unknown"
        if provider is None:
            provider = "unknown"
        labels = {"model": str(model), "provider": str(provider)}

        if cost_usd is not None and self._cost_counter is not None:
            with suppress(Exception):
                self._cost_counter.labels(**labels).inc(float(cost_usd))

        if total_tokens is not None and self._token_counter is not None:
            with suppress(Exception):
                self._token_counter.labels(**labels).inc(int(total_tokens))

        if latency_ms is not None and self._latency_hist is not None:
            with suppress(Exception):
                self._latency_hist.labels(**labels).observe(float(latency_ms) / 1000.0)

    def record_span(self, span: Any) -> None:
        if not self.enabled or span is None:
            return
        if getattr(span, "name", None) != "llm.generate":
            return

        attrs = getattr(span, "attributes", {}) or {}
        model = attrs.get("llm.model") or "unknown"
        provider = attrs.get("llm.provider") or "unknown"
        total_tokens = attrs.get("llm.total_tokens")
        if total_tokens is None:
            prompt = attrs.get("llm.prompt_tokens")
            completion = attrs.get("llm.completion_tokens")
            if prompt is not None or completion is not None:
                total_tokens = int(prompt or 0) + int(completion or 0)
        cost_usd = attrs.get("llm.cost_usd")
        latency_ms = attrs.get("llm.latency_ms")

        self.record_llm(
            model=str(model),
            provider=str(provider),
            cost_usd=float(cost_usd) if cost_usd is not None else None,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            latency_ms=float(latency_ms) if latency_ms is not None else None,
        )
