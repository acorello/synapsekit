"""SynapseKit observability dashboard — FastAPI server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SynapseKit Observability Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 24px;
        }
        h1 { color: #58a6ff; font-size: 1.6rem; margin-bottom: 8px; }
        .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 24px; }
        .badge {
            display: inline-block;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 0.75rem;
            color: #58a6ff;
            margin-left: 8px;
        }
        .metrics-row {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 28px;
        }
        .metric-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px 24px;
            min-width: 150px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #58a6ff;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #8b949e;
            margin-top: 4px;
        }
        section { margin-bottom: 32px; }
        section h2 {
            font-size: 1.1rem;
            color: #e6edf3;
            margin-bottom: 12px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 6px;
        }
        section h3 {
            font-size: 0.95rem;
            color: #e6edf3;
            margin: 12px 0 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        th {
            text-align: left;
            padding: 8px 12px;
            background: #161b22;
            color: #8b949e;
            font-weight: 600;
            border-bottom: 1px solid #30363d;
        }
        td {
            padding: 8px 12px;
            border-bottom: 1px solid #21262d;
            color: #c9d1d9;
        }
        tr:hover td { background: #1c2128; }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        .tag-ok { background: #0d4429; color: #3fb950; }
        .tag-warn { background: #3d2400; color: #d29922; }
        .tag-bad { background: #2d0101; color: #f85149; }
        .empty { color: #484f58; font-style: italic; padding: 16px 12px; }
        #refresh-badge {
            float: right;
            font-size: 0.78rem;
            color: #484f58;
        }
        .step-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }
        .step-label { font-size: 0.82rem; min-width: 140px; color: #8b949e; }
        .step-fill {
            height: 14px;
            background: #1f6feb;
            border-radius: 3px;
            min-width: 4px;
        }
        .step-val { font-size: 0.78rem; color: #8b949e; }
        .rag-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }
        .rag-panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
        }
        .rag-panel ul {
            list-style: none;
            display: grid;
            gap: 8px;
        }
        .rag-panel li {
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 10px 12px;
            color: #c9d1d9;
        }
        .rag-muted { color: #8b949e; font-size: 0.8rem; }
    </style>
</head>
<body>
    <h1>SynapseKit <span class="badge">Observability</span></h1>
    <p class="subtitle">
        Live dashboard — LLM traces, RAG metrics, alerts, ROI, agent timelines.
        <span id="refresh-badge">Auto-refreshing every 5s</span>
    </p>

    <div class="metrics-row" id="metric-cards">
        <!-- filled by JS -->
    </div>

    <section>
        <h2>Recent LLM Traces (last 100)</h2>
        <table id="traces-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Timestamp</th>
                    <th>Model</th>
                    <th>Prompt Tokens</th>
                    <th>Completion Tokens</th>
                    <th>Latency (ms)</th>
                    <th>Est. Cost (USD)</th>
                </tr>
            </thead>
            <tbody id="traces-body">
                <tr><td colspan="7" class="empty">Loading...</td></tr>
            </tbody>
        </table>
    </section>

    <section>
        <h2>RAG Pipeline Metrics</h2>
        <div id="rag-metrics">Loading...</div>
    </section>

    <section>
        <h2>RAG Alerts & Remediations</h2>
        <div id="rag-alerts">Loading...</div>
    </section>

    <section>
        <h2>Agent Execution Timeline</h2>
        <div id="agent-timeline">Loading...</div>
    </section>

    <script>
        function fmtTs(ts) {
            if (!ts) return '—';
            return new Date(ts * 1000).toLocaleTimeString();
        }
        function fmtScore(v) {
            if (v === null || v === undefined) return '—';
            const n = parseFloat(v);
            if (Number.isNaN(n)) return '—';
            const cls = n >= 0.8 ? 'tag-ok' : n >= 0.5 ? 'tag-warn' : 'tag-bad';
            return '<span class="tag ' + cls + '">' + n.toFixed(3) + '</span>';
        }
        function fmtNumber(v, digits = 2) {
            if (v === null || v === undefined) return '—';
            const n = Number(v);
            if (Number.isNaN(n)) return '—';
            return n.toFixed(digits);
        }
        function fmtMoney(v, digits = 4) {
            if (v === null || v === undefined) return '—';
            const n = Number(v);
            if (Number.isNaN(n)) return '—';
            return '$' + n.toFixed(digits);
        }

        async function refreshAll() {
            try {
                const [tracesResp, metricsResp] = await Promise.all([
                    fetch('/api/traces'),
                    fetch('/api/metrics'),
                ]);
                const traces = await tracesResp.json();
                const metrics = await metricsResp.json();
                renderMetricCards(metrics);
                renderTraces(traces);
                renderRagMetrics(metrics);
                renderRagAlerts(metrics);
                renderTimeline(traces);
            } catch (e) {
                console.error('Dashboard refresh error:', e);
            }
        }

        function renderMetricCards(m) {
            const cards = [
                { value: m.total_calls ?? 0, label: 'Total LLM Calls' },
                { value: (m.total_tokens ?? 0).toLocaleString(), label: 'Total Tokens' },
                { value: fmtMoney(m.total_cost_usd), label: 'Est. Cost (USD)' },
                { value: fmtNumber(m.avg_latency_ms, 0) + 'ms', label: 'Avg Latency' },
                { value: m.rag_evaluations ?? 0, label: 'RAG Evaluations' },
                { value: fmtMoney(m.total_rag_eval_cost_usd), label: 'RAG Eval Cost' },
                { value: m.total_rag_alerts ?? 0, label: 'RAG Alerts' },
                { value: fmtNumber(m.avg_rag_benefit_to_cost, 2), label: 'RAG ROI' },
            ];
            document.getElementById('metric-cards').innerHTML = cards.map(c =>
                '<div class="metric-card"><div class="metric-value">' + c.value + '</div><div class="metric-label">' + c.label + '</div></div>'
            ).join('');
        }

        function renderTraces(traces) {
            const tbody = document.getElementById('traces-body');
            if (!traces.length) {
                tbody.innerHTML = '<tr><td colspan="7" class="empty">No traces recorded yet.</td></tr>';
                return;
            }
            tbody.innerHTML = traces.map((t, i) =>
                '<tr>' +
                '<td>' + (i + 1) + '</td>' +
                '<td>' + fmtTs(t.timestamp) + '</td>' +
                '<td>' + (t.model || '—') + '</td>' +
                '<td>' + (t.input_tokens ?? '—') + '</td>' +
                '<td>' + (t.output_tokens ?? '—') + '</td>' +
                '<td>' + (t.latency_ms != null ? t.latency_ms.toFixed(1) : '—') + '</td>' +
                '<td>' + (t.cost_usd != null ? '$' + t.cost_usd.toFixed(6) : '—') + '</td>' +
                '</tr>'
            ).join('');
        }

        function renderRagMetrics(m) {
            const el = document.getElementById('rag-metrics');
            const rows = [
                ['Sample Rate', fmtNumber(m.rag_sample_rate, 2)],
                ['Evaluations', m.rag_evaluations ?? 0],
                ['Sampled Evaluations', m.rag_sampled_evaluations ?? 0],
                ['Skipped Evaluations', m.rag_skipped_evaluations ?? 0],
                ['Avg Recall', fmtScore(m.avg_rag_recall)],
                ['Avg Precision', fmtScore(m.avg_rag_precision)],
                ['Avg Relevance', fmtScore(m.avg_rag_relevance)],
                ['Avg Answer Quality', fmtScore(m.avg_rag_answer_quality)],
                ['Avg Retrieval Benefit', fmtScore(m.avg_rag_retrieval_benefit)],
                ['Avg Benefit / Cost', fmtNumber(m.avg_rag_benefit_to_cost, 2)],
                ['Total Eval Cost', fmtMoney(m.total_rag_eval_cost_usd)],
                ['Avg Eval Cost', fmtMoney(m.avg_rag_eval_cost_usd)],
                ['Total Alerts', m.total_rag_alerts ?? 0],
                ['RAG Trend', m.rag_quality_trend || '—'],
                ['Last Notes', m.rag_last_notes || '—'],
            ];
            el.innerHTML = '<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>' +
                rows.map(r => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('') +
                '</tbody></table>';
        }

        function renderRagAlerts(m) {
            const el = document.getElementById('rag-alerts');
            const alerts = m.rag_last_alerts || [];
            const suggestions = m.rag_last_suggestions || [];
            const alertList = alerts.length
                ? '<ul>' + alerts.map(a =>
                    '<li><strong>' + (a.metric || 'metric') + '</strong> · ' +
                    (a.severity || 'info') + '<br>' +
                    (a.message || '—') + '<br><span class="rag-muted">' +
                    (a.recommendation || '—') + '</span></li>'
                ).join('') + '</ul>'
                : '<p class="empty">No recent alerts.</p>';
            const suggestionList = suggestions.length
                ? '<ul>' + suggestions.map(s =>
                    '<li><strong>' + (s.metric || 'metric') + '</strong> · ' +
                    (s.action || '—') + '<br><span class="rag-muted">' +
                    (s.reason || '—') + '</span></li>'
                ).join('') + '</ul>'
                : '<p class="empty">No recent remediation suggestions.</p>';
            el.innerHTML =
                '<div class="rag-grid">' +
                '<div class="rag-panel"><h3>Latest Alerts</h3>' + alertList + '</div>' +
                '<div class="rag-panel"><h3>Latest Suggestions</h3>' + suggestionList + '</div>' +
                '</div>';
        }

        function renderTimeline(traces) {
            const el = document.getElementById('agent-timeline');
            if (!traces.length) {
                el.innerHTML = '<p class="empty">No agent steps recorded yet.</p>';
                return;
            }
            const maxLatency = Math.max(...traces.map(t => t.latency_ms || 0)) || 1;
            el.innerHTML = traces.slice(0, 20).map((t, i) => {
                const pct = Math.max(4, ((t.latency_ms || 0) / maxLatency) * 300);
                return '<div class="step-bar">' +
                    '<span class="step-label">' + (t.model || 'call-' + (i + 1)) + '</span>' +
                    '<div class="step-fill" style="width:' + pct + 'px"></div>' +
                    '<span class="step-val">' + (t.latency_ms != null ? t.latency_ms.toFixed(0) + 'ms' : '—') + '</span>' +
                    '</div>';
            }).join('');
        }

        refreshAll();
        setInterval(refreshAll, 5000);
    </script>
</body>
</html>
"""


def _build_trace_list(tracer: Any) -> list[dict[str, Any]]:
    """Convert TokenTracer records to a JSON-serialisable list."""
    from synapsekit.observability.tracer import COST_TABLE

    costs = COST_TABLE.get(tracer.model, {})
    records = tracer._records[-100:]  # last 100
    result = []

    for i, rec in enumerate(records):
        cost = rec.input_tokens * costs.get("input", 0.0) + rec.output_tokens * costs.get(
            "output", 0.0
        )
        # quality record aligned by index if available
        q_rec = None
        if i < len(tracer._quality_records):
            q_rec = tracer._quality_records[i]
        result.append(
            {
                "index": i + 1,
                "timestamp": q_rec.timestamp if q_rec else None,
                "model": tracer.model,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "latency_ms": rec.latency_ms,
                "cost_usd": cost,
            }
        )
    return result


def create_app(tracer: Any | None = None, rag_evaluator: Any | None = None) -> FastAPI:
    """Create the observability dashboard FastAPI app.

    Args:
        tracer: Optional ``TokenTracer`` instance. If None, a default one is created.
        rag_evaluator: Optional ``RAGEvaluator`` instance for sampled RAG metrics.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    from synapsekit.observability.tracer import COST_TABLE, TokenTracer

    app = FastAPI(title="SynapseKit Observability Dashboard")

    if tracer is None:
        tracer = TokenTracer(model="gpt-4o-mini", enabled=True)

    app.state.tracer = tracer
    app.state.rag_evaluator = rag_evaluator

    @app.get("/")
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/traces")
    def get_traces() -> JSONResponse:
        t = app.state.tracer
        return JSONResponse(_build_trace_list(t))

    @app.get("/api/metrics")
    def get_metrics() -> JSONResponse:
        t = app.state.tracer
        summary = t.summary()
        rag_summary = app.state.rag_evaluator.summary() if app.state.rag_evaluator else {}
        costs = COST_TABLE.get(t.model, {})
        records = t._records

        total_cost = sum(
            r.input_tokens * costs.get("input", 0.0) + r.output_tokens * costs.get("output", 0.0)
            for r in records
        )
        avg_latency = sum(r.latency_ms for r in records) / len(records) if records else 0.0

        def prefer(primary: Any, fallback: Any) -> Any:
            return fallback if primary is None else primary

        rag_eval_summary = rag_summary or {}
        rag_alerts = rag_eval_summary.get("alerts") or {}

        return JSONResponse(
            {
                "total_calls": summary["calls"],
                "total_tokens": summary["total_tokens"],
                "total_input_tokens": summary["total_input_tokens"],
                "total_output_tokens": summary["total_output_tokens"],
                "total_cost_usd": round(total_cost, 6),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_faithfulness": summary["avg_faithfulness"],
                "avg_relevancy": summary["avg_relevancy"],
                "quality_trend": summary["quality_trend"],
                "total_quality_records": len(t._quality_records),
                "rag_evaluations": prefer(rag_eval_summary.get("evaluations"), summary.get("rag_evaluations", 0)),
                "rag_sample_rate": rag_eval_summary.get("sample_rate"),
                "rag_sampled_evaluations": prefer(
                    rag_eval_summary.get("sampled_evaluations"), summary.get("rag_evaluations", 0)
                ),
                "rag_skipped_evaluations": rag_eval_summary.get("skipped_evaluations"),
                "avg_rag_recall": prefer(rag_eval_summary.get("avg_recall"), summary.get("avg_rag_recall")),
                "avg_rag_precision": prefer(rag_eval_summary.get("avg_precision"), summary.get("avg_rag_precision")),
                "avg_rag_relevance": prefer(rag_eval_summary.get("avg_relevance"), summary.get("avg_rag_relevance")),
                "avg_rag_answer_quality": prefer(
                    rag_eval_summary.get("avg_answer_quality"), summary.get("avg_rag_answer_quality")
                ),
                "avg_rag_retrieval_benefit": prefer(
                    rag_eval_summary.get("avg_retrieval_benefit"), summary.get("avg_rag_retrieval_benefit")
                ),
                "avg_rag_benefit_to_cost": prefer(
                    rag_eval_summary.get("avg_benefit_to_cost"), summary.get("avg_rag_benefit_to_cost")
                ),
                "total_rag_eval_cost_usd": prefer(
                    rag_eval_summary.get("total_eval_cost_usd"), summary.get("total_rag_eval_cost_usd")
                ),
                "avg_rag_eval_cost_usd": prefer(
                    rag_eval_summary.get("avg_eval_cost_usd"), summary.get("avg_rag_eval_cost_usd")
                ),
                "total_rag_alerts": prefer(rag_alerts.get("total"), summary.get("total_rag_alerts", 0)),
                "rag_alerts_by_metric": rag_alerts.get("by_metric", {}),
                "rag_alerts_by_severity": rag_alerts.get("by_severity", {}),
                "rag_quality_trend": summary.get("rag_quality_trend"),
                "rag_last_notes": prefer(rag_eval_summary.get("last_notes"), summary.get("last_notes")),
                "rag_last_sample_key": rag_eval_summary.get("last_sample_key"),
                "rag_last_question": rag_eval_summary.get("last_question"),
                "rag_last_alerts": rag_eval_summary.get("last_alerts", []),
                "rag_last_suggestions": rag_eval_summary.get("last_suggestions", []),
            }
        )

    return app
