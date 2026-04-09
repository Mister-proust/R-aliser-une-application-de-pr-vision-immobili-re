# ✅ GitHub Actions + K3S Workflow - Implementation Complete!

## 🎉 What's Been Done

Your automated CI/CD pipeline is ready! Here's what was created:

### 📁 Files Created
1. ✅ `.github/workflows/deploy-k3s.yml` - Main GitHub Actions workflow
2. ✅ `scripts/deploy-k3s.sh` - Manual deployment helper script
3. ✅ `k3s/GITHUB_ACTIONS_SETUP.md` - Detailed setup & configuration guide
4. ✅ `k3s/GITHUB_WORKFLOW_README.md` - Workflow overview & quick reference

### 📝 Files Modified
1. ✅ `k3s/manifests/deployment-mcp.yaml` - Added GHCR support + command override
2. ✅ `k3s/manifests/deployment-fastapi.yaml` - Added GHCR support + command override
3. ✅ `k3s/manifests/deployment-gradio.yaml` - Added GHCR support + command override
4. ✅ `k3s/manifests/services.yaml` - Added sessionAffinity for stateful MCP
5. ✅ `Dockerfile` - Cleaned up HEALTHCHECK & EXPOSE directives

---

## 🚀 Next Steps (Required Actions)

### 📋 Priority 1: Create GitHub Secrets (DO THIS FIRST!)

**Step 1: Extract Kubeconfig from Datalab Server**
```bash
# SSH into datalab
ssh p4g3@datalab.myconnectech.fr
# Password: GRETA@2026

# Display kubeconfig
cat /etc/rancher/k3s/k3s.yaml

# Then modify ONLY the 'server' line:
# Original: server: https://127.0.0.1:6443
# Change to: server: https://datalab.myconnectech.fr:6443

# Copy the entire modified content
```

**Step 2: Generate GHCR Token**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: `GHCR_TOKEN`
4. Scopes: ✅ read:packages + ✅ write:packages
5. Generate & copy the token

**Step 3: Create 3 GitHub Secrets**
1. Go to: https://github.com/mister-proust/r-aliser-une-application-de-pr-vision-immobili-re
2. Click: Settings → Secrets and variables → Actions
3. Create these secrets:

| Name | Value |
|------|-------|
| `K3S_KUBECONFIG` | Paste kubeconfig from Step 1 |
| `GHCR_TOKEN` | Paste token from Step 2 |
| `ENV_SECRETS` | See below ↓ |

**ENV_SECRETS Format:**
```
# Copy these values from your local .env file
# ⚠️ NEVER paste real secrets into documentation or commit to git!
HF_API_KEY=<your_huggingface_token_from_local_env>
MISTRAL_API_KEY=<your_mistral_api_key_from_local_env>
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=<your_langsmith_api_key_from_local_env>
LANGSMITH_PROJECT=<your_project_name_from_local_env>

# K3S network configuration
MCP_SERVER_HOST=mcp-service.expert-immo.svc.cluster.local
MCP_SERVER_PORT=8001
```
(One variable per line, KEY=VALUE format - Copy real values ONLY from your local .env)

---

### 📋 Priority 2: Trigger First Deployment

**Option A: Automatic (Recommended)**
```bash
git push origin main
# GitHub Actions will automatically start the workflow
```

**Option B: Manual Trigger**
1. Go to GitHub repository
2. Click: Actions tab
3. Select: "Build & Deploy to K3S + GHCR"
4. Click: "Run workflow"
5. Monitor in real-time

---

### 📋 Priority 3: Monitor & Verify

**Check Workflow Progress**
1. Go to: Actions tab
2. Follow the two jobs:
   - `build-and-push` (builds & pushes Docker image)
   - `deploy-to-k3s` (deploys to K3S on datalab)

**Verify Deployment on K3S**
```bash
# SSH to datalab
ssh p4g3@datalab.myconnectech.fr

# Check pods
kubectl get pods -n expert-immo

# Should see 3 mcp-server, 3 fastapi-server, 3 gradio-agent pods

# Watch deployment
kubectl get pods -n expert-immo -w
```

**Test Access Points**
- FastAPI: http://datalab.myconnectech.fr:30800
- Gradio: http://datalab.myconnectech.fr:30786
- Docs: http://datalab.myconnectech.fr:30800/docs

---

## 📚 Documentation Reference

### For Setup Issues
👉 Read: `k3s/GITHUB_ACTIONS_SETUP.md`
- Step-by-step configuration
- Troubleshooting guide
- Secret management procedures

### For Workflow Overview
👉 Read: `k3s/GITHUB_WORKFLOW_README.md`
- Architecture explanation
- Deployment stages
- Access points & operations

### For Manual Deployment
👉 Use: `scripts/deploy-k3s.sh`
```bash
./scripts/deploy-k3s.sh "ghcr.io/mister-proust/r-aliser-une-application-de-pr-vision-immobili-re:main-abc123"
```

---

## 🎯 Workflow Architecture

```
Your Code Push to Main
    ↓
GitHub Actions Triggers
    ├─ Job 1: build-and-push
    │  ├─ Build Docker image
    │  ├─ Push to GHCR (ghcr.io/mister-proust/...)
    │  └─ Tag: main-<commit-sha>
    │
    └─ Job 2: deploy-to-k3s (waits for Job 1)
       ├─ Setup kubeconfig
       ├─ Create GHCR secret
       ├─ Create environment secrets
       ├─ Deploy 3 services:
       │  ├─ MCP Server (port 8001)
       │  ├─ FastAPI (port 8000 → 30800)
       │  └─ Gradio (port 7860 → 30786)
       └─ Verify all ready
    
Deployment Complete ✅
Access via nodeport on datalab
```

---

## 💡 Key Features Implemented

✅ **Single Image, 3 Services**
- One Dockerfile, three different entry points via K3S `command` overrides

✅ **GHCR Integration**
- Automatically builds and pushes to GitHub Container Registry
- Secure credential management via GitHub Secrets

✅ **Stateful MCP Service**
- `sessionAffinity: ClientIP` ensures same-client requests hit same pod
- Important for MCP's stateful protocol

✅ **Automated Secrets**
- GHCR credentials injected at deploy time
- Environment variables from GitHub Secrets

✅ **Semantic Versioning**
- Git tags trigger release builds (e.g., `v1.0.0`)
- Automatic tag management: `main-<sha>`, `v<semver>`

✅ **Zero-Downtime Deployment**
- Init containers wait for dependencies (FastAPI/Gradio wait for MCP)
- Health checks verify service readiness

---

## ⚙️ Configuration Summary

| Component | Configuration | Value |
|-----------|---|---|
| **Image Registry** | GHCR | ghcr.io/mister-proust/... |
| **Namespace** | K3S | expert-immo |
| **MCP Service** | Type | ClusterIP (internal) |
| **MCP Affinity** | SessionAffinity | ClientIP (10s timeout) |
| **FastAPI** | NodePort | 30800 |
| **Gradio** | NodePort | 30786 |
| **Replicas** | All services | 3 (load balanced) |
| **Kubeconfig** | Source | /etc/rancher/k3s/k3s.yaml |

---

## 🆘 Quick Troubleshooting

### "ImagePullBackOff" Error
→ Check GHCR_TOKEN secret is correct
```bash
kubectl describe pod <pod-name> -n expert-immo
```

### Pod Stuck in "Pending"
→ Check node resources
```bash
kubectl describe nodes
```

### Workflow Not Triggering
→ Verify .github/workflows/deploy-k3s.yml is on main branch
```bash
git push origin main
```

### MCP Service Not Responding
→ Verify sessionAffinity is set
```bash
kubectl get svc mcp-service -n expert-immo -o json | grep sessionAffinity
```

See `k3s/GITHUB_ACTIONS_SETUP.md` for detailed troubleshooting.

---

## 📌 Important Notes

⚠️ **Kubeconfig is sensitive** - Keep K3S_KUBECONFIG secret safe
- Only share GitHub Secrets securely
- Never commit kubeconfig to version control

⚠️ **MCP is stateful** - `sessionAffinity: ClientIP` is critical
- Ensures same-client requests hit same pod
- Do not remove from services.yaml

⚠️ **Environment variables** - Update ENV_SECRETS if you change .env
- Redeploy after updating:
  ```bash
  git push origin main
  ```

---

## ✨ Success Criteria

You'll know it's working when:
1. ✅ GitHub Actions workflow completes successfully
2. ✅ Image appears in GHCR (ghcr.io/mister-proust/...)
3. ✅ 3 pods of each service running: `kubectl get pods -n expert-immo`
4. ✅ FastAPI responds: http://datalab.myconnectech.fr:30800
5. ✅ Gradio responds: http://datalab.myconnectech.fr:30786
6. ✅ MCP service healthy: `kubectl logs mcp-server-* -n expert-immo`

---

## 📞 Need Help?

| Question | Answer |
|----------|--------|
| How do I create GitHub Secrets? | See `k3s/GITHUB_ACTIONS_SETUP.md` Step 2 |
| How do I extract kubeconfig? | See `k3s/GITHUB_ACTIONS_SETUP.md` Step 1.2 |
| What if workflow fails? | Check GitHub Actions logs, then k3s/GITHUB_ACTIONS_SETUP.md troubleshooting |
| Can I deploy manually? | Yes: `./scripts/deploy-k3s.sh <image-tag>` |
| How do I monitor K3S pods? | SSH to datalab, then `kubectl get pods -n expert-immo` |

---

## 🎊 Ready to Deploy!

**You're all set!** Follow the "Next Steps (Required Actions)" section above and you'll have:
- ✅ Automated Docker builds on every push
- ✅ Automatic deployments to K3S
- ✅ 3 load-balanced microservices
- ✅ Stateful MCP routing
- ✅ GHCR image registry
- ✅ Secure secret management

**Happy deploying! 🚀**

---

**Created**: April 9, 2026
**Workflow Status**: ✅ Ready for first deployment
**Documentation**: Complete - see `k3s/GITHUB_ACTIONS_SETUP.md` & `k3s/GITHUB_WORKFLOW_README.md`
