# Kubernetes Deployment (Beginner Friendly)

This folder deploys your project to local Kubernetes (Docker Desktop Kubernetes) with:

- `agent-service` (FastAPI backend)
- `streamlit-app` (UI)
- `prometheus` (metrics collector)
- `grafana` (dashboard UI)

## What Each File Does

- `namespace.yaml`
  - Creates namespace `agent-platform` to keep all resources together.

- `agent-configmap.yaml`
  - Non-secret backend settings (ports, sqlite paths, monitoring flags).

- `streamlit-configmap.yaml`
  - Sets `AGENT_URL` so Streamlit talks to `agent-service` inside cluster.

- `agent-secret.example.yaml`
  - Template for secrets (API keys and `USER_AUTH_SECRET`).
  - Do not commit real secrets.

- `agent-deployment.yaml`
  - Runs backend container and adds health probes (`/healthz`, `/readyz`).

- `agent-service.yaml`
  - Internal Kubernetes service for backend on port `8000`.

- `streamlit-deployment.yaml`
  - Runs Streamlit container with health probe.

- `streamlit-service.yaml`
  - Exposes Streamlit via NodePort `30501`.

- `prometheus-configmap.yaml`
  - Prometheus scrape config for `agent-service:8000/metrics`.

- `prometheus-deployment.yaml`
  - Runs Prometheus pod.

- `prometheus-service.yaml`
  - Exposes Prometheus UI via NodePort `30900`.

- `grafana-datasource-configmap.yaml`
  - Auto-configures Grafana to use Prometheus as default data source.

- `grafana-deployment.yaml`
  - Runs Grafana pod.

- `grafana-service.yaml`
  - Exposes Grafana UI via NodePort `30300`.

- `kustomization.yaml`
  - Lets you apply all manifests in one command.

## 1) Prerequisites

- Docker Desktop with Kubernetes enabled.
- `kubectl` installed and pointing to Docker Desktop cluster.

Verify:

```bash
kubectl config current-context
kubectl get nodes
```

## 2) Build Local Images

Run from repo root:

```bash
docker build -f docker/Dockerfile.service -t agent-service-toolkit/agent-service:local .
docker build -f docker/Dockerfile.app -t agent-service-toolkit/streamlit-app:local .
```

## 3) Create Secret

Use your real values:

```bash
kubectl create namespace agent-platform
kubectl -n agent-platform create secret generic agent-secrets \
  --from-literal=USER_AUTH_SECRET=replace-with-long-random-secret \
  --from-literal=OPENAI_API_KEY=replace-if-used \
  --from-literal=GROQ_API_KEY=replace-if-used \
  --from-literal=LANGSMITH_API_KEY=replace-if-used
```

If the namespace already exists, ignore that message.

## 4) Deploy Everything

```bash
kubectl apply -k k8s
```

Check status:

```bash
kubectl -n agent-platform get pods
kubectl -n agent-platform get svc
```

## 5) Access URLs

- Streamlit: `http://localhost:30501`
- Prometheus: `http://localhost:30900`
- Grafana: `http://localhost:30300`
  - default login: `admin` / `admin`

## 6) Useful Commands

See recent logs:

```bash
kubectl -n agent-platform logs deploy/agent-service --tail=200
kubectl -n agent-platform logs deploy/prometheus --tail=200
kubectl -n agent-platform logs deploy/grafana --tail=200
```

Delete stack:

```bash
kubectl delete -k k8s
kubectl delete namespace agent-platform
```

## Notes

- This setup is for learning and local testing.
- `emptyDir` storage means pod restart can clear local runtime data.
- For production, use persistent volumes, real secret management, and ingress/TLS.
