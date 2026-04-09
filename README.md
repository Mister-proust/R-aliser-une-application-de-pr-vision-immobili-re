# Expert Immobilier IA

Application intelligente de prévision et d'analyse du marché immobilier français, basée sur les données **Demandes de Valeurs Foncières (DVF)**.

Ce projet combine un modèle de Machine Learning performant (XGBoost), une interface conversationnelle pilotée par LLM (Mistral), et une API REST robuste.

---

## Fonctionnalités Clés

- **Agent IA Expert** : Discutez avec un expert immobilier via une interface Gradio. L'agent peut estimer des biens, rechercher des transactions historiques et géolocaliser des adresses.
- **Modèle de Prédiction** : Estimation précise basée sur la surface, le type de bien, le nombre de pièces, la densité de population et la localisation géographique.
- **Exploration de Données DVF** : Recherche directe dans la base de données SQL des transactions réelles (S1 2025).
- **Géocodage & Analyse Géo** : Intégration de l'API Adresse pour localiser les biens et enrichissement des données avec les caractéristiques des communes.
- **Interface Web & API** : Une application FastAPI complète pour l'inférence et la visualisation.

---

## Architecture du Projet

Le système est structuré en trois piliers :

1.  **Serveur MCP** (`src/mcp_server/`) : Le cœur fonctionnel qui expose les outils (Estimation, SQL, Géocodage) via le protocole *Model Context Protocol*.
2.  **Agentia (Gradio)** (`src/agentia/`) : L'interface utilisateur intelligente qui orchestre les outils MCP via LangChain et Mistral AI.
3.  **App Web (FastAPI)** (`src/app/`) : Une interface web classique et des endpoints API pour une utilisation traditionnelle.

---

## ⚙️ Installation et Configuration

### 1. Prérequis
Assurez-vous d'avoir Python 3.10+ et l'outil `uv` installé.

```bash
pip install uv
uv sync
```

### 2. Configuration de l'environnement
Copiez le fichier `template.env` vers `.env` et renseignez vos clés :
```bash
cp template.env .env
```
Variables requises :
- `MISTRAL_API_KEY` : Votre clé Mistral AI.
- `MCP_SERVER_PORT` : Port du serveur MCP (défaut : 8001).
- `FASTAPI_PORT` : Port de l'application FastAPI (défaut : 8000).
- `GRADIO_PORT` : Port de l'interface Gradio (défaut : 7860).

### 3. Préparation des données (DVC)
Le projet utilise DVC pour la gestion des données volumineuses et des modèles.
```bash
uv run dvc repro
```
*Note : Assurez-vous d'avoir les fichiers sources `ValeursFoncieres-2025-S1.txt` et `communes-france-2025.csv` dans le dossier `data/`.*

---

## Lancement de l'Application

Pour une expérience complète (Agent IA), suivez cet ordre :

1.  **Démarrer le Serveur MCP** (Indispensable pour l'agent) :
    ```bash
    uv run src/mcp_server/server.py
    ```
2.  **Démarrer l'Agent Expert (Gradio)** :
    ```bash
    uv run src/agentia/main.py
    ```
    *Accès : http://127.0.0.1:7860/*

3.  **Démarrer l'Application Web (FastAPI)** :
    ```bash
    uv run src/app/main.py
    ```
    *Accès : http://127.0.0.1:8000/ (Documentation : /docs)*

---

## Structure du Projet

```text
.
├── data/                # Données brutes et modèles versionnés (DVC)
├── scripts/             # Pipelines de nettoyage et d'entraînement
│   ├── clean_dvf.py     # Nettoyage des données sources
│   ├── model.py         # Entraînement XGBoost
│   └── transform_csv_to_db.py # Création de la base SQL
├── src/
│   ├── agentia/         # Agent IA (Mistral + LangChain + Gradio)
│   ├── mcp_server/      # Outils exposés via MCP
│   └── app/             # Application Web FastAPI
├── dvc.yaml             # Définition du pipeline de données
└── .env                 # Configuration des clés API
```

---

## Outils MCP Disponibles

Le Serveur MCP expose les outils suivants pour l'Agent IA :

### 1. **estimate_property** - Estimation de Propriété
```
estimate_property(
  commune: str,           # Nom de commune ou code INSEE
  type_bien: str,         # "Maison" ou "Appartement"
  surface: int,           # Surface habitable en m²
  rooms: int = 1,         # Nombre de pièces principales
  surface_terrain: int = 0,  # Surface du terrain en m² (optionnel)
  type_voie: str = "Rue"  # Type de voie
)
```
**Retour** : Estimation du prix en euros basée sur le modèle XGBoost + caractéristiques communes.

### 2. **diagnostic_quartier** - Diagnostic de Quartier
```
diagnostic_quartier(
  code_postal: str,        # Code postal français (5 chiffres)
  commune_name: str,       # Nom de commune (avec validation)
  latitude: float,         # (optionnel) Pour localisation par coordonnées
  longitude: float
)
```
**Retour** : Statistiques du quartier basées sur les transactions DVF :
- Nombre de transactions (Jan-Juin 2025)
- Prix moyen et médian
- Nombre de pièces moyen
- Distribution des types de bien
- Tendance marché (hausse/baisse)
- Caractéristiques géographiques (densité, etc.)

### 3. **geocoding_search** - Géocodage Direct
```
geocoding_search(
  q: str,                # Adresse à rechercher
  index: str = "address",  # "address", "poi", ou "parcel"
  limit: int = 5,        # Nombre de résultats
  postcode: str,         # Filtrer par code postal (optionnel)
  city: str,             # Filtrer par ville (optionnel)
  type: str              # Type d'adresse: "housenumber", "street", "municipality"
)
```
**Retour** : Coordonnées GPS et label de l'adresse trouvée.

### 4. **reverse_geocoding** - Géocodage Inverse
```
reverse_geocoding(
  lon: float,           # Longitude
  lat: float,           # Latitude
  index: str = "address",
  limit: int = 1
)
```
**Retour** : Adresse correspondant aux coordonnées GPS.

### 5. **get_database_schema** - Schéma de la BD
```
get_database_schema()
```
**Retour** : Liste des tables et colonnes disponibles dans la base de données DVF (Transactions, Communes).

### 6. **execute_sql** - Requête SQL Personnalisée
```
execute_sql(query: str)  # Query SELECT uniquement (limitation de sécurité)
```
**Retour** : Résultats de la requête SELECT.

**Limitations** :
- Seules les requêtes `SELECT` sont autorisées
- Validation des codes postaux et communes absents
- Si aucun résultat et présence de WHERE, affiche message explicite de couverture

---

## Sécurité & Validation des Entrées

### Validations Implémentées

1. **Codes Postaux** : Regex `^[0-9]{5}$` (5 chiffres obligatoires)
   - Rejeté : "75 001", "7500", "75001a"
   - Accepté : "75001", "13013"

2. **Noms de Communes** : Regex `^[a-zA-Z\s\-àâäçèéêëîïôùûüœæ]{1,100}$`
   - Rejeté : "Paris@", "Saint-Rémy123"
   - Accepté : "Paris", "Saint-Rémy", "Lyon"

3. **Type de Bien** : Enumération stricte ["Maison", "Appartement"]
   - Rejeté : "maison", "maison et garage"
   - Accepté : "Maison", "Appartement"

4. **Requêtes SQL** : Limitation à `SELECT` uniquement
   - Bloque : INSERT, UPDATE, DELETE, DROP
   - Avec analyse comportement pour détection injection

### Gestion des Erreurs

Tous les outils retournent des messages d'erreur explicites :
- Code postal non couvert : *"Aucune commune trouvée avec le code postal 99999..."*
- Lieu absent de la BD : *"Le département XX n'est pas couvert par ma base de données..."*
- Format invalide : *"Code postal invalide '75-001'. Format attendu: 5 chiffres..."*

---

## Déploiement Kubernetes (K3S)

Pour déployer l'application sur un cluster K3S, consultez le guide détaillé : [k3s/DEPLOYMENT_GUIDE.md](k3s/DEPLOYMENT_GUIDE.md)

### Démarrage Rapide

```bash
# 1. Vérifier K3S installé
sudo k3s --version

# 2. Exporter kubeconfig
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# 3. Builder les images Docker
docker build -t expert-immo-fastapi:latest -f Dockerfile.fastapi .
docker build -t expert-immo-mcp:latest -f Dockerfile.mcp .
docker build -t expert-immo-gradio:latest -f Dockerfile.gradio .

# 4. Créer namespace et appliquer manifests
kubectl create namespace expert-immo
kubectl apply -f k3s/manifests/ -n expert-immo

# 5. Vérifier déploiement
kubectl get pods -n expert-immo -w

# 6. Accéder aux services (option 1: port-forward)
kubectl port-forward svc/gradio-service 7860:7860 -n expert-immo &
kubectl port-forward svc/fastapi-service 8000:8000 -n expert-immo &
# Accès: http://localhost:7860 (Gradio) et http://localhost:8000 (FastAPI)

# Ou (option 2: NodePort direct)
# Accès: http://<node-ip>:30786 (Gradio) et http://<node-ip>:30800 (FastAPI)
```

### Architecture K3S

```
Namespace: expert-immo
├── Deployment: mcp-server (port 8001, interne)
│   └── Service: mcp-service (ClusterIP)
├── Deployment: fastapi-server (port 8000, dépend de mcp)
│   └── Service: fastapi-service (NodePort:30800)
├── Deployment: gradio-agent (port 7860, dépend de mcp)
│   └── Service: gradio-service (NodePort:30786)
└── PersistentVolumeClaim: shared DB + model (5Gi)
```

### Configuration K3S

- **Storage** : Local-path provisioner (K3S default)
- **Networking** : Services communiquent via DNS interne (mcp-service:8001)
- **Init Containers** : FastAPI et Gradio attendent MCP avant de démarrer
- **Resource Limits** : Définis pour chaque pod

Pour détails complets (variables d'env, secrets, scaling, troubleshooting) → [k3s/DEPLOYMENT_GUIDE.md](k3s/DEPLOYMENT_GUIDE.md)

---

## Conteneurisation Docker

### Images Docker

**3 images modulaires** pour déploiement indépendant :

1. **expert-immo-fastapi** (Dockerfile.fastapi)
   - FastAPI web server (port 8000)
   - Dépend de: Database + Modèle ML

2. **expert-immo-mcp** (Dockerfile.mcp)
   - MCP Server (port 8001)
   - Dépend de: Database + Modèle ML
   - Requis par: FastAPI + Gradio

3. **expert-immo-gradio** (Dockerfile.gradio)
   - Gradio Agent Interface (port 7860)
   - Dépend de: MCP Server

### Build Local

```bash
# Builder les images
docker build -t expert-immo-fastapi:latest -f Dockerfile.fastapi .
docker build -t expert-immo-mcp:latest -f Dockerfile.mcp .
docker build -t expert-immo-gradio:latest -f Dockerfile.gradio .

# Vérifier
docker images | grep expert-immo
```

### Lancer en Docker Compose (optionnel)

```bash
# Créer un docker-compose.yml pour orchestration locale
docker-compose up -d

# Accès:
# FastAPI: http://localhost:8000
# Gradio: http://localhost:7860
# MCP: http://localhost:8001
```

---

## Troubleshooting & Diagnostiqué

### Application Locale

**Problème** : Import error "No module named 'agentia'"
```bash
# Solution
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
uv run src/mcp_server/server.py
```

**Problème** : "Base de données introuvable"
```bash
# Regénérer la BD
uv run dvc repro
# Vérifier présence du fichier
ls -la data/models/xgb_pipeline.pkl
```

**Problème** : "Code postal / commune non couverts"
- Normal : La BD DVF ne couvre que certains départements
- Solution : Utiliser un autre code postal ou consulter la couverture dans les logs

### Déploiement K3S

**Problème** : Pod en "Pending"
```bash
kubectl describe pod <pod-name> -n expert-immo
# Vérifier: PVC, limites ressources, image availability
```

**Problème** : "Connection refused" entre services
```bash
# Tester DNS et connectivity
kubectl exec -it pod/mcp-server-xxx -n expert-immo -- \
  nslookup mcp-service && echo "✅ DNS OK"
```

**Problème** : MCP ou Agent crash au démarrage
```bash
# Vérifier logs
kubectl logs -f pod/mcp-server-xxx -n expert-immo

# Contrôler env vars
kubectl exec pod/mcp-server-xxx -n expert-immo -- env
```

Pour assistance complète → [k3s/DEPLOYMENT_GUIDE.md](k3s/DEPLOYMENT_GUIDE.md)

---

## Détails Techniques

- **Modèle** : XGBoost Regressor (Pipeline avec OrdinalEncoder et StandardScaler).
- **Couverture** : Le modèle et la base de données couvrent actuellement une sélection de départements français (vérification automatique de la couverture lors des requêtes).
- **Limites** : Les données transactions couvrent la période du 2 janvier au 30 juin 2025.
- **Sécurité** : Les requêtes SQL sont limitées aux commandes `SELECT` uniquement. Validations sur entrées utilisateur présentes.

---
*Réalisé dans le cadre de la formation Développeur IA - Promotion 2026.*
