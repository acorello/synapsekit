# SynapseKit Observability Helm Chart

Deploy Prometheus + Grafana with a preloaded SynapseKit dashboard.

## Install

```bash
helm install synapsekit-observability ./assets/helm/synapsekit-observability
```

## Configure

Update `values.yaml` if your SynapseKit metrics service is different:

```yaml
synapsekitMetrics:
  service:
    name: synapsekit-metrics
    port: 8000
```

Prometheus scrapes the configured service. Grafana is pre-provisioned with:
- Prometheus datasource
- SynapseKit dashboard (from `assets/grafana/synapsekit-observe-dashboard.json`)

## Access

```bash
kubectl port-forward svc/synapsekit-observability-grafana 3000:3000
```

Open http://localhost:3000 (admin/admin by default).
