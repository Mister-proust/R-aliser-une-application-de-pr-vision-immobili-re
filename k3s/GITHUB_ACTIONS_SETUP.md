# K3S Deployment Setup Guide - GitHub Actions + Secrets Configuration

## Overview
This guide walks you through setting up the GitHub Actions workflow to build Docker images, push to GHCR (GitHub Container Registry), and deploy to K3S on the datalab server.

## Prerequisites
- GitHub repository with write permissions
- K3S cluster running on datalab server
- SSH access to datalab
- Git repository cloned locally

## Step 1: Create GitHub Secrets

### 1.1 Generate GHCR Token (Personal Access Token)

1. Go to GitHub: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Fill in:
   - **Token name**: `GHCR_TOKEN` (or similar)
   - **Expiration**: 90 days (or longer)
   - **Scopes**: Check these boxes:
     - ✅ `read:packages` - to read packages
     - ✅ `write:packages` - to write packages
     - ✅ `delete:packages` - (optional, for cleanup)

4. Click **"Generate token"**
5. **Copy the token immediately** (you won't see it again!)

### 1.2 Extract Kubeconfig from Datalab

1. SSH :
   ```bash
   ssh 
   
   ```

2. Display kubeconfig:
   ```bash
   cat /etc/rancher/k3s/k3s.yaml
   ```

3. Copy the **entire output** (it's a YAML file)

4. Modify the kubeconfig:
   - Change `127.0.0.1` → `xxxxxxxxxxxxxxx`
   - Or use the actual K3S server IP if different
   

### 1.3 Create Environment Secrets File

Prepare your `.env` variables in single-line format (your actual variables):
```
HF_API_KEY=
MISTRAL_API_KEY=
LANGSMITH_TRACING=
LANGSMITH_ENDPOINT=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
MCP_SERVER_HOST=
MCP_SERVER_PORT=
```

## Step 2: Add Secrets to GitHub Repository

1. Go to your GitHub repository

2. Click **Settings** → **Secrets and variables** → **Actions**

3. Create these secrets by clicking **"New repository secret"**:

### Secret 1: K3S_KUBECONFIG
- **Name**: `K3S_KUBECONFIG`
- **Value**: Paste the entire kubeconfig content from Step 1.2
  - **Encode it in base64** first:
    ```bash
    # macOS/Linux:
    cat /path/to/k3s.yaml | base64
    
    # Windows PowerShell:
    [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Content k3s.yaml -Raw))) | Set-Clipboard
    ```
  - Or paste the raw YAML directly (GitHub will handle encoding)

### Secret 2: GHCR_TOKEN
- **Name**: `GHCR_TOKEN`
- **Value**: Paste the Personal Access Token from Step 1.1

### Secret 3: ENV_SECRETS
- **Name**: `ENV_SECRETS`
- **Value**: Paste environment variables in format (from your .env):
  - One variable per line
  - Use `KEY=VALUE` format
  - **Note**: No DATABASE_URL needed (you use SQLite locally)

## Step 3: Verify Workflow Trigger

1. Go to your GitHub repo: **Actions** tab
2. You should see the workflow **"Build & Deploy to K3S + GHCR"**
3. The workflow triggers on:
   - ✅ Push to `main` branch
   - ✅ Git tags matching `v*.*.*` (e.g., `v1.0.0`)

## Step 4: Manual Deployment (Testing)

### Option A: Deploy from GitHub Actions UI
1. Go to **Actions** → Select **"Build & Deploy to K3S + GHCR"**
2. Click **"Run workflow"** → **"Branch: main"** → **"Run workflow"**
3. Monitor logs in real-time

### Option B: Deploy via Command Line (Local Testing)

First, ensure you have `kubectl` installed and configured:
```bash
# Set kubeconfig
export KUBECONFIG=/path/to/k3s.yaml

# Verify connectivity
kubectl cluster-info

# Deploy with helper script
./scripts/deploy-k3s.sh "ghcr.io/mister-proust/r-aliser-une-application-de-pr-vision-immobili-re:main-abc1234" g3-immo
```

## Step 5: Monitor Deployment

### Check Pod Status:
```bash
# List all pods in g3-immo namespace
kubectl get pods -n g3-immo

# Watch pods in real-time
kubectl get pods -n g3-immo -w

# Describe a specific pod
kubectl describe pod <pod-name> -n g3-immo

# View pod logs
kubectl logs <pod-name> -n g3-immo
```

### Check Service Status:
```bash
# List services
kubectl get svc -n g3-immo

# Get external IPs/NodePorts
kubectl get svc -n g3-immo -o wide
```

### Verify Image Pull:
```bash
# Check if GHCR secret is mounted
kubectl get secret -n g3-immo

# Inspect pod events (shows image pull errors)
kubectl describe pod <pod-name> -n g3-immo | grep -A 10 Events
```

## Step 6: Access Deployed Services

### FastAPI:
```
http://datalab.myconnectech.fr:30800
http://datalab.myconnectech.fr:30800/docs (Swagger UI)
```

### Gradio:
```
http://datalab.myconnectech.fr:30786
```

### MCP Service (Internal Only):
```
mcp-service.g3-immo.svc.cluster.local:8001
```

## Troubleshooting

### Issue: ImagePullBackOff Error
**Symptom**: Pod stuck in `ImagePullBackOff` state
```bash
kubectl describe pod mcp-server-xyz -n g3-immo
# Error: failed to pull image "ghcr.io/...": authentication required
```

**Solution**:
1. Verify `ghcr-secret` exists:
   ```bash
   kubectl get secret ghcr-secret -n g3-immo
   ```

2. Recreate the secret:
   ```bash
   kubectl create secret docker-registry ghcr-secret \
     --docker-server=ghcr.io \
     --docker-username=<github-username> \
     --docker-password=<GHCR_TOKEN> \
     --docker-email=<github-email> \
     -n g3-immo --dry-run=client -o yaml | kubectl apply -f -
   ```

3. Restart pods:
   ```bash
   kubectl rollout restart deployment/mcp-server -n g3-immo
   kubectl rollout restart deployment/fastapi-server -n g3-immo
   kubectl rollout restart deployment/gradio-agent -n g3-immo
   ```

### Issue: Pod Stuck in Pending State
**Symptom**: Pod never becomes `Running`
```bash
kubectl describe pod <pod-name> -n g3-immo
# Events: Insufficient memory, Insufficient CPU, etc.
```

**Solution**:
1. Check node resources:
   ```bash
   kubectl describe nodes
   ```

2. Reduce resource requests in deployments:
   ```bash
   # Open deployment YAML and reduce `resources.requests`
   kubectl edit deployment mcp-server -n g3-immo
   ```

### Issue: MCP StatefulnessProblem
**Symptom**: Requests from same client going to different pods
```bash
# Verify sessionAffinity is set
kubectl get svc mcp-service -n g3-immo -o jsonpath='{.spec.sessionAffinity}'
# Should output: ClientIP
```

**Solution**:
- This is already configured in `services.yaml` with `sessionAffinity: ClientIP`
- Redeploy services:
  ```bash
  kubectl apply -f k3s/manifests/services.yaml -n g3-immo
  ```

## Workflow Execution Details

### Build Stage (Job: build-and-push)
1. Checkout code
2. Setup Docker Buildx (multi-platform builds)
3. Login to GHCR
4. Extract metadata (tags, labels)
5. Build and push Docker image
   - Tags: `main-<commit-sha>`, `v<semver>` (for releases)
6. Output image digest

### Deploy Stage (Job: deploy-to-k3s)
1. Checkout code
2. Setup kubeconfig from secrets
3. Verify kubectl connectivity
4. Create/ensure namespace `g3-immo` exists
5. Create GHCR docker-registry secret
6. Create generic environment secrets
7. Deploy 3 services:
   - MCP Server (port 8001, internal)
   - FastAPI (port 8000, NodePort 30800)
   - Gradio (port 7860, NodePort 30786)
8. Wait for deployments to reach ready state
9. Print deployment summary

## Maintenance

### Manual Secret Management

#### Update GHCR Token:
```bash
# Regenerate token on GitHub Settings
# Then update secret:
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<new-token> \
  --docker-email=<email> \
  -n g3-immo \
  --dry-run=client -o yaml | kubectl apply -f -
```

#### Update Environment Variables:
```bash
# Update secret-immo-env
kubectl create secret generic expert-immo-env \
  --from-literal=MISTRAL_API_KEY=new_key \
  --from-literal=MCP_SERVER_HOST=... \
  -n g3-immo \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up changes
kubectl rollout restart deployment/fastapi-server -n expert-immo
```

### Clean Up Old Images from GHCR
```bash
# List packages
curl -H "Authorization: token <GHCR_TOKEN>" \
  https://api.github.com/user/packages?package_type=container

# Delete old image tags (not recommended for production)
```

## References
- [GitHub Actions Docker Build & Push](https://github.com/docker/build-push-action)
- [K3S Documentation](https://docs.k3s.io/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [GHCR Docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

## Questions?
For more help, check:
- GitHub Actions logs: Repository → Actions → Workflow run
- K3S logs on datalab: `sudo journalctl -u k3s -f`
- This repo: k3s/DEPLOYMENT_GUIDE.md
