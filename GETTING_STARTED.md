# 🚀 Guide de Déploiement - Expert Immobilier IA

**Namespace K3S utilisé :** `g3-immo`

---

## 📋 Flux de Déploiement Simplifié

```
1. Prérequis           → K3S + kubectl + données + images Docker
2. Configuration       → .env secrets et variables
3. Déploiement manuel  → kubectl apply manifests (développement local)
OU
3. Déploiement auto    → Git push → GitHub Actions → K3S (production)
4. Vérification        → Pods running → Services accessible
5. Accès               → FastAPI / Gradio / MCP
```

---

## 🔧 Prérequis

### 1. **K3S Installé et Running**
```bash
# Vérifier K3S
sudo k3s --version
sudo systemctl status k3s

# Configurer kubectl locale (si besoin)
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl cluster-info
```

### 2. **Données et Modèles Disponibles**
```bash
# Générer les données via DVC
dvc repro
# Cela crée :
# - data/clean_dvf.csv
# - data/communes-france-2025.csv
# - data/models/xgb_pipeline.pkl
```

### 3. **Images Docker Construites**
```bash
# Option A : Construire localement
docker build -t g3-immo-mcp:latest \
  -f Dockerfile --target mcp_server .

docker build -t g3-immo-fastapi:latest \
  -f Dockerfile --target fastapi_app .

docker build -t g3-immo-gradio:latest \
  -f Dockerfile --target gradio_agent .

# Option B : Utiliser GHCR (GitHub Container Registry)
# → Automatique via GitHub Actions
```

---

## 🚀 Déploiement Local (Développement)

### Étape 1 : Créer le Namespace
```bash
kubectl create namespace g3-immo
```

### Étape 2 : Configurer les Secrets et ConfigMaps
```bash
# Créer les secrets sensibles
kubectl create secret generic g3-env \
  --from-literal=MISTRAL_API_KEY='your-key-here' \
  --from-literal=LANGSMITH_API_KEY='optional-key' \
  -n g3-immo

# Les ConfigMaps sont appliquées avec les manifests
```

### Étape 3 : Appliquer les Manifests **dans cet ordre**
```bash
# 1. PVC et ConfigMap d'abord
kubectl apply -f k3s/manifests/pvc-config.yaml -n g3-immo

# 2. MCP Server (dépendance pour les autres)
kubectl apply -f k3s/manifests/deployment-mcp.yaml -n g3-immo

# 3. FastAPI et Gradio (attendent MCP)
kubectl apply -f k3s/manifests/deployment-fastapi.yaml -n g3-immo
kubectl apply -f k3s/manifests/deployment-gradio.yaml -n g3-immo

# 4. Services (exposition des ports)
kubectl apply -f k3s/manifests/services.yaml -n g3-immo

# Ou tout d'un coup (après dépendances)
kubectl apply -f k3s/manifests/ -n g3-immo
```

### Étape 4 : Vérifier les Deployments
```bash
# Attendre que tous les pods soient Ready
watch kubectl get pods -n g3-immo -o wide

# Sortie attendue :
# NAME                              READY   STATUS    RESTARTS
# mcp-server-xyz                    1/1     Running   0
# fastapi-server-abc                1/1     Running   0
# gradio-agent-def                  1/1     Running   0

# Sortir : Ctrl+C
```

### Étape 5 : Accéder aux Services

**Option A : Port-forward local** (Plus simple)
```bash
# Terminal 1
kubectl port-forward svc/mcp-service 8001:8001 -n g3-immo

# Terminal 2
kubectl port-forward svc/fastapi-service 8000:8000 -n g3-immo

# Terminal 3
kubectl port-forward svc/gradio-service 7860:7860 -n g3-immo

# Accès :
# - FastAPI:  http://localhost:8000/docs
# - Gradio:   http://localhost:7860
# - MCP:      http://localhost:8001 (interne)
```

**Option B : NodePort** (Accès direct depuis le serveur K3S)
```bash
# Récupérer l'IP du nœud K3S
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')

# Accès :
# - FastAPI:  http://$NODE_IP:30800/docs
# - Gradio:   http://$NODE_IP:30786
# - MCP:      Interne uniquement
```

---

## 🤖 Déploiement Automatisé (GitHub Actions)

### Prérequis GitHub

1. **Secrets à configurer** dans `Settings → Secrets and variables → Actions` :

| Nom | Description |
|-----|-------------|
| `K3S_KUBECONFIG` | Contenu complet de `/etc/rancher/k3s/k3s.yaml` |
| `GHCR_TOKEN` | Token GitHub avec accès `packages:write` |
| `ENV_SECRETS` | Variables d'env au format `KEY1=val1\nKEY2=val2\n...` |

2. **Fichier `.env` complet** (référence) :
```env
MISTRAL_API_KEY=your-mistral-key
LANGSMITH_API_KEY=optional-langsmith-key
```

### Automatisation

**À chaque push sur `main` :**

```bash
git add .
git commit -m "Update deployment"
git push origin main
```

Cela déclenche :
1. ✅ Construction des images Docker
2. ✅ Push vers GHCR
3. ✅ Déploiement automatique sur K3S
4. ✅ Vérification des pods

Suivi dans : `.github/workflows/deploy-k3s.yml`

---

## 🔍 Dépannage

### Pods en "Pending" ou "ImagePullBackOff"

```bash
# Voir les erreurs détaillées
kubectl describe pod <pod-name> -n g3-immo

# Vérifier les images locales
docker images | grep g3-immo

# Reconstruire si besoin
docker build -t g3-immo-mcp:latest -f Dockerfile --target mcp_server .
```

### "Connection refused" au port-forward

```bash
# Vérifier service endpoints
kubectl get endpoints -n g3-immo

# Voir les logs du pod
kubectl logs deployment/mcp-server -n g3-immo
kubectl logs deployment/fastapi-server -n g3-immo
kubectl logs deployment/gradio-agent -n g3-immo
```

### MCP Pas Accessible depuis FastAPI/Gradio

```bash
# Test DNS interne
kubectl exec -it deployment/fastapi-server -n g3-immo -- \
  nslookup mcp-service.g3-immo.svc.cluster.local

# Test connectivité
kubectl exec -it deployment/fastapi-server -n g3-immo -- \
  wget -O- http://mcp-service:8001/
```

### Erreurs "Database Not Found"

```bash
# Vérifier PVC montée
kubectl get pvc -n g3-immo

# Vérifier données dans pod
kubectl exec deployment/mcp-server -n g3-immo -- \
  ls -la /app/agentia/bdd/
```

---

## 📊 Monitoring

```bash
# Voir les pods et services
kubectl get pods,svc -n g3-immo

# Ressources utilisées
kubectl top pods -n g3-immo

# Logs d'un déploiement entier
kubectl logs deployment/mcp-server -n g3-immo -f

# Logs d'un pod spécifique
kubectl logs <pod-name> -n g3-immo -f
```

---

## 🗑️ Cleanup

```bash
# Supprimer tout le namespace
kubectl delete namespace g3-immo

# Ou sélectif
kubectl delete deployment --all -n g3-immo
kubectl delete svc --all -n g3-immo
kubectl delete pvc --all -n g3-immo
kubectl delete secret --all -n g3-immo
```

---

## 📝 Notes Techniques

| Aspect | Valeur |
|--------|--------|
| **Namespace** | `g3-immo` |
| **Replicas** | 3 (MCP, FastAPI, Gradio) |
| **Storage** | 5GB local-path PVC |
| **DNS interne** | `mcp-service.g3-immo.svc.cluster.local:8001` |
| **Ports | FastAPI(8000), Gradio(7860), MCP(8001) |
| **NodePorts** | FastAPI(30800), Gradio(30786) |
| **Init Container** | Attendre MCP avant FastAPI/Gradio |
| **Session Affinity** | MCP (stateful) |

---

## 📂 Fichiers Clés

```
k3s/
├── manifests/
│   ├── pvc-config.yaml         ← PVC 5GB + ConfigMap
│   ├── deployment-mcp.yaml     ← MCP Server (port 8001)
│   ├── deployment-fastapi.yaml ← FastAPI (port 8000)
│   ├── deployment-gradio.yaml  ← Gradio (port 7860)
│   └── services.yaml           ← Services + NodePorts
├── DEPLOYMENT_GUIDE.md         ⚠️ OBSOLÈTE - Siehe GETTING_STARTED.md
└── GITHUB_ACTIONS_SETUP.md     ⚠️ OBSOLÈTE - Siehe GETTING_STARTED.md

.github/workflows/
└── deploy-k3s.yml             ← Automatisation GitHub Actions
```

---

## ❓ Questions Fréquentes

**Q : Est-ce que je peux scaler les replicas ?**  
A : Oui, sauf pour MCP (doit rester `replicas: 1`). Voir scaling ci-dessus.

**Q : Mes données ne sont pas à jour ?**  
A : Relancer `dvc repro` et redéployer avec `kubectl rollout restart`.

**Q : Comment accéder à distance (SSH via datalab) ?**

```bash
ssh -L 8000:localhost:8000 -L 7860:localhost:7860 \
  p4g1@datalab.myconnectech.fr \
  'kubectl port-forward svc/fastapi-service 8000:8000 -n g3-immo & \
   kubectl port-forward svc/gradio-service 7860:7860 -n g3-immo & \
   sleep infinity'

# Puis accéder localement sur http://localhost:8000 et http://localhost:7860
```

---

**Dernière mise à jour :** avril 2026  
**Version :** 1.0 (Guide maître unifié)
