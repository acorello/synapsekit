"""Tests for the observability UI server endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from synapsekit.cli.ui_server import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"status": "ok"}


def test_health_endpoint_status_key_is_string(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert isinstance(resp.json()["status"], str)


# ---------------------------------------------------------------------------
# /api/metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_200(client: TestClient) -> None:
    resp = client.get("/api/metrics")
    assert resp.status_code == 200


def test_metrics_endpoint_has_required_keys(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    required = {
        "total_calls",
        "total_tokens",
        "total_cost_usd",
        "avg_latency_ms",
        "rag_evaluations",
        "total_rag_eval_cost_usd",
        "total_rag_alerts",
        "avg_rag_benefit_to_cost",
    }
    for key in required:
        assert key in data, f"Missing key: {key!r}"


def test_metrics_total_calls_is_int(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert isinstance(data["total_calls"], int)


def test_metrics_total_cost_usd_is_float_or_int(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert isinstance(data["total_cost_usd"], (float, int))


def test_metrics_avg_latency_ms_is_numeric(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert isinstance(data["avg_latency_ms"], (float, int))


def test_metrics_empty_tracer_returns_zero_calls(client: TestClient) -> None:
    data = client.get("/api/metrics").json()
    assert data["total_calls"] == 0


# ---------------------------------------------------------------------------
# /api/traces
# ---------------------------------------------------------------------------


def test_traces_endpoint_returns_list(client: TestClient) -> None:
    resp = client.get("/api/traces")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_traces_empty_by_default(client: TestClient) -> None:
    data = client.get("/api/traces").json()
    assert data == []


def test_traces_after_recording(client: TestClient) -> None:
    # Seed a record into the tracer
    tracer = client.app.state.tracer  # type: ignore[attr-defined]
    tracer.record(input_tokens=10, output_tokens=5, latency_ms=123.4)
    data = client.get("/api/traces").json()
    assert len(data) >= 1
    trace = data[0]
    assert "input_tokens" in trace
    assert "output_tokens" in trace
    assert "latency_ms" in trace


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------


def test_root_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "SynapseKit" in resp.text
    assert "RAG Alerts & Remediations" in resp.text


# ---------------------------------------------------------------------------
# create_app accepts optional tracer
# ---------------------------------------------------------------------------


def test_create_app_with_custom_tracer() -> None:
    from synapsekit.observability.tracer import TokenTracer

    tracer = TokenTracer(model="gpt-4o")
    tracer.record(input_tokens=100, output_tokens=50, latency_ms=200.0)
    app = create_app(tracer=tracer)
    c = TestClient(app)
    data = c.get("/api/metrics").json()
    assert data["total_calls"] == 1
    assert data["total_tokens"] == 150


class _FakeRAGEvaluator:
    def summary(self) -> dict[str, object]:
        return {
            "evaluations": 3,
            "sample_rate": 0.5,
            "sampled_evaluations": 2,
            "skipped_evaluations": 1,
            "avg_recall": 0.81,
            "avg_precision": 0.72,
            "avg_relevance": 0.76,
            "avg_answer_quality": 0.83,
            "avg_retrieval_benefit": 0.79,
            "avg_benefit_to_cost": 117.5,
            "total_eval_cost_usd": 0.0042,
            "avg_eval_cost_usd": 0.0021,
            "alerts": {
                "total": 1,
                "by_metric": {"precision": 1},
                "by_severity": {"warning": 1},
            },
            "last_notes": "use a reranker",
            "last_sample_key": "sample-1",
            "last_question": "What is RAG?",
            "last_alerts": [
                {
                    "metric": "precision",
                    "severity": "warning",
                    "message": "Noisy retrieval detected.",
                    "recommendation": "Enable reranking.",
                    "value": 0.42,
                    "threshold": 0.65,
                }
            ],
            "last_suggestions": [
                {
                    "metric": "precision",
                    "action": "Enable reranking.",
                    "reason": "Low precision from recent samples.",
                }
            ],
        }


def test_metrics_with_rag_evaluator_surface_roi_and_alerts() -> None:
    from synapsekit.observability.tracer import TokenTracer

    tracer = TokenTracer(model="gpt-4o")
    tracer.record(input_tokens=12, output_tokens=8, latency_ms=42.0)
    app = create_app(tracer=tracer, rag_evaluator=_FakeRAGEvaluator())
    c = TestClient(app)
    data = c.get("/api/metrics").json()
    assert data["rag_evaluations"] == 3
    assert data["rag_sample_rate"] == 0.5
    assert data["total_rag_alerts"] == 1
    assert data["rag_last_question"] == "What is RAG?"
    assert data["rag_last_alerts"][0]["metric"] == "precision"
    assert data["rag_last_suggestions"][0]["action"] == "Enable reranking."
