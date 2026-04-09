#!/bin/bash
# Deploy script to apply K3S manifests with dynamic image tagging
# Usage: ./deploy-k3s.sh <image-tag> [namespace]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_TAG="${1:-}"
NAMESPACE="${2:-expert-immo}"
MANIFEST_DIR="k3s/manifests"
TEMP_DIR=$(mktemp -d)

# Functions
log_info() {
    echo -e "${GREEN}ℹ️  $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

cleanup() {
    rm -rf "$TEMP_DIR"
}

# Validate inputs
if [ -z "$IMAGE_TAG" ]; then
    log_error "Usage: $0 <image-tag> [namespace]"
fi

# Trap cleanup on exit
trap cleanup EXIT

log_info "Deploying to namespace: $NAMESPACE"
log_info "Image tag: $IMAGE_TAG"

# Check kubectl connectivity
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
fi

# Ensure namespace exists
kubectl get namespace "$NAMESPACE" &> /dev/null || \
    kubectl create namespace "$NAMESPACE"
log_info "Namespace $NAMESPACE ready"

# Apply manifests with image tag substitution
declare -a DEPLOYMENTS=(
    "deployment-mcp.yaml"
    "deployment-fastapi.yaml"
    "deployment-gradio.yaml"
)

for deployment in "${DEPLOYMENTS[@]}"; do
    if [ ! -f "$MANIFEST_DIR/$deployment" ]; then
        log_error "Manifest not found: $MANIFEST_DIR/$deployment"
    fi
    
    # Create temp file with substituted image tag
    temp_file="$TEMP_DIR/$deployment"
    sed "s|IMAGE_PLACEHOLDER|$IMAGE_TAG|g" "$MANIFEST_DIR/$deployment" > "$temp_file"
    
    # Apply manifest
    kubectl apply -f "$temp_file" --namespace="$NAMESPACE"
    log_info "Applied $deployment"
done

# Apply services
kubectl apply -f "$MANIFEST_DIR/services.yaml" --namespace="$NAMESPACE"
log_info "Applied services.yaml"

# Apply PVC and ConfigMap
kubectl apply -f "$MANIFEST_DIR/pvc-config.yaml" --namespace="$NAMESPACE"
log_info "Applied pvc-config.yaml"

# Wait for deployments to be ready (with timeout)
log_info "Waiting for deployments to be ready..."

for deployment in mcp-server fastapi-server gradio-agent; do
    log_info "Checking deployment: $deployment"
    if kubectl rollout status deployment/"$deployment" \
        --namespace="$NAMESPACE" \
        --timeout=300s; then
        log_info "✓ Deployment $deployment is ready"
    else
        log_warn "⚠️  Deployment $deployment did not reach ready state within timeout"
    fi
done

# Print deployment summary
echo ""
echo -e "${GREEN}=== Deployment Summary ===${NC}"
kubectl get pods,svc --namespace="$NAMESPACE"

echo ""
echo -e "${GREEN}=== Access Points ===${NC}"
echo "FastAPI:  http://datalab.myconnectech.fr:30800"
echo "Gradio:   http://datalab.myconnectech.fr:30786"
echo "MCP (internal): mcp-service.expert-immo.svc.cluster.local:8001"

log_info "✓ Deployment completed successfully!"
