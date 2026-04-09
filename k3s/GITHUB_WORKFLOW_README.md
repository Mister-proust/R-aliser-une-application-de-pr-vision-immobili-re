# GitHub Actions + K3S Deployment Workflow

## 📋 Overview

This workflow automates the complete CI/CD pipeline for your real estate prediction application:

1. **Build Stage** (`build-and-push`)
   - Triggers on commits to `main` or Git tags (`v*.*.*`)
   - Builds Docker image using the root Dockerfile
   - Pushes to GHCR (GitHub Container Registry) with semantic versioning

2. **Deploy Stage** (`deploy-to-k3s`)
   - Deploys all 3 microservices to K3S on datalab:
     - **MCP Server** (Port 8001, internal only)
     - **FastAPI** (Port 8000, exposed via NodePort 30800)
     - **Gradio UI** (Port 7860, exposed via NodePort 30786)
   - Manages secrets, namespace, and load balancing

## 📁 Files Created/Modified

### **GitHub Actions Workflow**
- [`.github/workflows/deploy-k3s.yml`](.github/workflows/deploy-k3s.yml) - Main CI/CD pipeline

### **K3S Kubernetes Manifests** (Modified)
- [`k3s/manifests/deployment-mcp.yaml`](k3s/manifests/deployment-mcp.yaml)
  - ✅ Added `imagePullSecrets` for GHCR
  - ✅ Added `command` override for MCP entry point
  - ✅ Changed replicas: 1 → 3
  - ✅ Changed image to `IMAGE_PLACEHOLDER`

- [`k3s/manifests/deployment-fastapi.yaml`](k3s/manifests/deployment-fastapi.yaml)
  - ✅ Added `imagePullSecrets` for GHCR
  - ✅ Added `command` override for FastAPI entry point
  - ✅ Changed replicas: 1 → 3
  - ✅ Changed image to `IMAGE_PLACEHOLDER`

- [`k3s/manifests/deployment-gradio.yaml`](k3s/manifests/deployment-gradio.yaml)
  - ✅ Added `imagePullSecrets` for GHCR
  - ✅ Added `command` override for Gradio entry point
  - ✅ Changed replicas: 1 → 3
  - ✅ Changed image to `IMAGE_PLACEHOLDER`

- [`k3s/manifests/services.yaml`](k3s/manifests/services.yaml)
  - ✅ Added `sessionAffinity: ClientIP` to mcp-service
  - ✅ Added `sessionAffinityConfig` for stateful MCP routing

### **Docker Image** (Modified)
- [`Dockerfile`](Dockerfile)
  - ✅ Removed hardcoded `HEALTHCHECK` (now in K3S deployments)
  - ✅ Removed `EXPOSE` directives (now in K3S manifests)
  - ✅ Added documentation about multi-service usage

### **Helper Scripts** (Created)
- [`scripts/deploy-k3s.sh`](scripts/deploy-k3s.sh) - Manual deployment script
  - Applies manifests with dynamic image tag substitution
  - Includes error handling and colored output
  - Usage: `./scripts/deploy-k3s.sh <image-tag> [namespace]`

### **Documentation** (Created)
- [`k3s/GITHUB_ACTIONS_SETUP.md`](k3s/GITHUB_ACTIONS_SETUP.md) - Complete setup guide
  - Step-by-step GitHub Secrets configuration
  - Kubeconfig extraction from datalab
  - Deployment verification & troubleshooting
  - Access points & maintenance procedures

## 🚀 Quick Start

### 1. Initial Setup (One-time)
```bash
# Read the complete setup guide
cat k3s/GITHUB_ACTIONS_SETUP.md

# Follow Step 1-2 to create GitHub Secrets:
# - K3S_KUBECONFIG
# - GHCR_TOKEN
# - ENV_SECRETS
```

### 2. Trigger Deployment
**Option A: Automatic (Recommended)**
```bash
# Push to main branch
git push origin main

# Or create a release tag
git tag v1.0.0
git push origin v1.0.0

# GitHub Actions will automatically build & deploy
```

**Option B: Manual Trigger**
1. Go to your GitHub repo → **Actions** tab
2. Select "Build & Deploy to K3S + GHCR" workflow
3. Click "Run workflow" button
4. Monitor progress in real-time

**Option C: Local Deployment (Testing)**
```bash
# Set your kubeconfig
export KUBECONFIG=/path/to/k3s.yaml

# Deploy with script
./scripts/deploy-k3s.sh "ghcr.io/mister-proust/r-aliser-une-application-de-pr-vision-immobili-re:main-abc123"

# Monitor
kubectl get pods -n expert-immo -w
```

## 📊 Workflow Stages

### Stage 1: Build & Push (`build-and-push`)
**Triggers**: Push to main + Git tags
```
Checkout Code
    ↓
Setup Docker Buildx
    ↓
Login to GHCR
    ↓
Extract Metadata (tags, labels)
    ↓
Build & Push Docker Image
    ↓
Output: ghcr.io/mister-proust/.../main-<SHA>
```

**Image Tags**:
- For `main` branch: `main-<commit-sha>` (e.g., `main-a1b2c3d`)
- For release tags: `v1.0.0`, `1.0`, `latest`

### Stage 2: Deploy to K3S (`deploy-to-k3s`)
**Triggers**: After successful build on `main` branch push
```
Checkout Code
    ↓
Setup Kubeconfig from Secrets
    ↓
Verify k8s Connectivity
    ↓
Ensure Namespace Exists (expert-immo)
    ↓
Create/Update GHCR Registry Secret
    ↓
Create/Update Environment Secrets
    ↓
Deploy MCP Server
    ↓ (waits for ready)
Deploy FastAPI Service
    ↓
Deploy Gradio Service
    ↓
Apply Services with SessionAffinity
    ↓
Verify All Deployments Ready
    ↓
Print Access Points
```

## 🔧 Key Features

### ✅ Multi-Service from Single Image
- One Docker image, 3 different entry points
- Controlled via `command` overrides in K3S deployments

### ✅ Semantic Versioning
- Automatic tagging: `main-<SHA>`, `v<semver>`
- Git tags trigger release builds

### ✅ Stateful MCP Service
- `sessionAffinity: ClientIP` ensures requests from same client go to same pod
- 10 second timeout for connection persistence

### ✅ Automated Secrets Management
- GHCR credentials from GitHub Secrets
- Environment variables injected at deploy time
- Kubeconfig tunneled securely

### ✅ Fail-Safe Deployment
- Kubectl connectivity verification
- Namespace auto-creation
- Rollout status checks with timeouts
- Init containers wait for MCP server

## 📊 Access Points

After successful deployment:

| Service | Type | URL |
|---------|------|-----|
| **FastAPI** | NodePort | http://datalab.myconnectech.fr:30800 |
| **FastAPI Docs** | NodePort | http://datalab.myconnectech.fr:30800/docs |
| **Gradio** | NodePort | http://datalab.myconnectech.fr:30786 |
| **MCP** | ClusterIP | mcp-service.expert-immo.svc.cluster.local:8001 |

## 🐛 Troubleshooting

### Check Workflow Logs
```bash
# GitHub Actions UI: https://github.com/.../actions
# Or via CLI: gh run list --workflow deploy-k3s.yml
```

### Monitor K3S Pods
```bash
# SSH to datalab
ssh p4g3@datalab.myconnectech.fr

# Check pods
kubectl get pods -n expert-immo
kubectl describe pod <pod-name> -n expert-immo

# View logs
kubectl logs mcp-server-xyz -n expert-immo
kubectl logs fastapi-server-abc -n expert-immo
kubectl logs gradio-agent-def -n expert-immo
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| `ImagePullBackOff` | GHCR secret missing | Recreate secret, check token |
| `Pending` | Low resources | Check node capacity or reduce limits |
| `CrashLoopBackOff` | App error | Check logs: `kubectl logs ...` |
| `Stateless MCP` | SessionAffinity not set | Reapply services.yaml |

See [`k3s/GITHUB_ACTIONS_SETUP.md`](k3s/GITHUB_ACTIONS_SETUP.md) for detailed troubleshooting.

## 📝 Configuration

### Environment Variables (in `ENV_SECRETS`)
```
# Copy these values from your local .env file
# ⚠️ NEVER commit secrets to git or paste in documentation!
HF_API_KEY=<copy_from_local_env>
MISTRAL_API_KEY=<copy_from_local_env>
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=<copy_from_local_env>
LANGSMITH_PROJECT=<copy_from_local_env>
MCP_SERVER_HOST=mcp-service.expert-immo.svc.cluster.local
MCP_SERVER_PORT=8001
```

### Resource Limits (K3S Deployments)
| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-----------|----------------|--------------|
| MCP | 250m | 500m | 512Mi | 1Gi |
| FastAPI | 200m | 500m | 256Mi | 1Gi |
| Gradio | 200m | 800m | 512Mi | 2Gi |

## 🔄 Manual Operations

### Rollback to Previous Image
```bash
kubectl set image deployment/fastapi-server \
  fastapi-server=ghcr.io/.../v1.0.0 \
  -n expert-immo
```

### Scale Deployments
```bash
kubectl scale deployment/fastapi-server --replicas=5 -n expert-immo
```

### Update Environment Variables
```bash
kubectl create secret generic expert-immo-env \
  --from-literal=MISTRAL_API_KEY=new_key \
  --dry-run=client -o yaml | kubectl apply -f -
  
# Restart pods to pick up changes
kubectl rollout restart deployment/fastapi-server -n expert-immo
```

### View Deployment History
```bash
kubectl rollout history deployment/mcp-server -n expert-immo
```

## 📚 Related Documentation
- [K3S Deployment Guide](k3s/DEPLOYMENT_GUIDE.md) - Original K3S setup
- [GitHub Actions Setup](k3s/GITHUB_ACTIONS_SETUP.md) - Detailed setup instructions
- [Docker Build & Push Action](https://github.com/docker/build-push-action)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

## ✨ Next Steps

1. **Create GitHub Secrets** (follow Setup Guide)
2. **Test Workflow** (push to main or trigger manually)
3. **Monitor Deployment** (watch Actions tab + kubectl)
4. **Verify Access** (visit http://datalab.myconnectech.fr:30800)
5. **Setup Monitoring** (optional: Prometheus/Grafana)
6. **Document Changes** (update this README as needed)

---

**Last Updated**: April 9, 2026
**Maintainer**: Your Team
**Status**: ✅ Ready for deployment
