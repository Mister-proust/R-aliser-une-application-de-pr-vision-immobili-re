# K3S Deployment Guide - Expert Immobilier IA

## Prerequisites

1. **K3S Installation**: Ensure K3S is installed on your server
   ```bash
   # Check K3S version
   sudo k3s --version
   
   # Start K3S if not running
   sudo systemctl start k3s
   ```

2. **kubectl Access**: Configure kubeconfig for local access
   ```bash
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
   
   # Verify connectivity
   kubectl cluster-info
   ```

3. **Docker Images**: Build images locally or have them available in a registry
   ```bash
   # Build images locally for K3S (runs docker directly)
   docker build -t expert-immo-fastapi:latest -f Dockerfile.fastapi .
   docker build -t expert-immo-mcp:latest -f Dockerfile.mcp .
   docker build -t expert-immo-gradio:latest -f Dockerfile.gradio .
   ```

4. **Data & Models**: Ensure these are available (DVC managed)
   ```bash
   dvc repro  # Generate clean_dvf.csv, communes data, and xgb_pipeline.pkl
   ```

---

## Deployment Steps

### 1. Create Namespace
```bash
kubectl create namespace expert-immo
```

### 2. Update Secrets (Optional but Recommended)
If using Kubernetes Secrets for sensitive data (instead of ConfigMap):
```bash
kubectl create secret generic expert-immo-secrets \
  --from-literal=MISTRAL_API_KEY='your-key-here' \
  -n expert-immo
```

### 3. Apply Manifests
```bash
# Apply in order of dependency (PVC -> ConfigMap -> Deployments -> Services)
kubectl apply -f k3s/manifests/pvc-config.yaml -n expert-immo
kubectl apply -f k3s/manifests/deployment-mcp.yaml -n expert-immo
kubectl apply -f k3s/manifests/deployment-fastapi.yaml -n expert-immo
kubectl apply -f k3s/manifests/deployment-gradio.yaml -n expert-immo
kubectl apply -f k3s/manifests/services.yaml -n expert-immo

# Or apply all at once
kubectl apply -f k3s/manifests/ -n expert-immo
```

### 4. Verify Deployment
```bash
# Check pod status (wait for all to be Running and 1/1 Ready)
watch kubectl get pods -n expert-immo -o wide

# View logs of MCP (should start first)
kubectl logs -f deployment/mcp-server -n expert-immo

# View logs of FastAPI (may show "waiting for MCP..." initially)
kubectl logs -f deployment/fastapi-server -n expert-immo

# View logs of Gradio
kubectl logs -f deployment/gradio-agent -n expert-immo
```

### 5. Access Services

**Option A: Using NodePort (Direct K3S node access)**
```bash
# Get node IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')

# Access services
# FastAPI Docs: http://$NODE_IP:30800/docs
# Gradio Interface: http://$NODE_IP:30786
```

**Option B: Using kubectl port-forward (Local access via tunnel)**
```bash
# Forward FastAPI port
kubectl port-forward svc/fastapi-service 8000:8000 -n expert-immo &

# Forward Gradio port
kubectl port-forward svc/gradio-service 7860:7860 -n expert-immo &

# Access services
# FastAPI: http://localhost:8000/docs
# Gradio: http://localhost:7860
```

**Option C: Using Port-Forward from Remote Machine (datalab)**
```bash
# From local machine, tunnel through datalab SSH
ssh -L 8000:localhost:8000 -L 7860:localhost:7860 p4g1@datalab.myconnectech.fr \
  'kubectl port-forward svc/fastapi-service 8000:8000 -n expert-immo & \
   kubectl port-forward svc/gradio-service 7860:7860 -n expert-immo & \
   sleep infinity'

# Access services
# FastAPI: http://localhost:8000/docs
# Gradio: http://localhost:7860
```

---

## Common Issues & Troubleshooting

### Issue: Pods stuck in "Pending" or "ImagePullBackOff"

**Cause**: Image not found or PVC not bound.

**Solution**:
```bash
# Check pod events for error details
kubectl describe pod deployment/mcp-server -n expert-immo

# Ensure images are available locally
docker images | grep expert-immo

# Rebuild images if needed
docker build -t expert-immo-mcp:latest -f Dockerfile.mcp .
```

### Issue: "Connection refused" when accessing services

**Cause**: Service not yet ready or listening.

**Solution**:
```bash
# Check service status
kubectl get svc -n expert-immo

# Check if service has endpoints
kubectl get endpoints -n expert-immo

# View pod logs to see startup errors
kubectl logs deployment/mcp-server -n expert-immo
```

### Issue: MCP Server not accessible from FastAPI/Gradio

**Cause**: Network policy or DNS resolution issue.

**Solution**:
```bash
# Test DNS resolution from FastAPI pod
kubectl exec -it deployment/fastapi-server -n expert-immo -- nslookup mcp-service

# Test connectivity to MCP service
kubectl exec -it deployment/fastapi-server -n expert-immo -- wget -O- http://mcp-service:8001/
```

### Issue: Database "not found" errors

**Cause**: PVC not mounted correctly or data not initialized.

**Solution**:
```bash
# Check PVC status
kubectl get pvc -n expert-immo

# Verify mount in pod
kubectl exec deployment/mcp-server -n expert-immo -- ls -la /app/agentia/bdd/

# Ensure data files are in place (may need init job)
# For now, mount source data manually if needed
```

### Issue: Out of memory errors

**Cause**: Gradio or model loading exceeds resource limits.

**Solution**:
```bash
# Increase resource limits in deployment YAML
# Edit: k3s/manifests/deployment-gradio.yaml
# Increase limits.memory from 2Gi to 4Gi

kubectl apply -f k3s/manifests/deployment-gradio.yaml -n expert-immo
```

---

## Scaling & Management

### Scale replicas
```bash
# Horizontal scaling (increase FastAPI replicas)
kubectl scale deployment fastapi-server --replicas=2 -n expert-immo

# Note: MCP should remain replicas=1 (single DB)
```

### View pod resource usage
```bash
kubectl top pods -n expert-immo
```

### Restart a deployment
```bash
kubectl rollout restart deployment/mcp-server -n expert-immo
```

### View deployment history
```bash
kubectl rollout history deployment/fastapi-server -n expert-immo
```

---

## Cleanup

```bash
# Delete all resources
kubectl delete namespace expert-immo

# Or selective delete
kubectl delete deployment -n expert-immo --all
kubectl delete svc -n expert-immo --all
kubectl delete pvc -n expert-immo --all
```

---

## Notes

- **Storage**: K3S uses local-path provisioner by default (local node storage). For multi-node K3S, consider using a shared storage solution (NFS, Longhorn, etc.).
- **Secrets**: Update configMap MISTRAL_API_KEY in pvc-config.yaml or use Kubernetes Secrets for production.
- **Networking**: Pods communicate via K3S internal DNS (service-name:port). MCP is only accessible internally.
- **Init Containers**: FastAPI and Gradio wait for MCP to be ready before starting (via init container).
