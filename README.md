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

## Détails Techniques

- **Modèle** : XGBoost Regressor (Pipeline avec OrdinalEncoder et StandardScaler).
- **Couverture** : Le modèle et la base de données couvrent actuellement une sélection de départements français (vérification automatique de la couverture lors des requêtes).
- **Limites** : Les données transactions couvrent la période du 2 janvier au 30 juin 2025.
- **Sécurité** : Les requêtes SQL sont limitées aux commandes `SELECT` uniquement.

---

## Déploiement (Docker)

Construire l'image :
```bash
docker build -t expert-immo:latest .
```
Lancer le conteneur :
```bash
docker run -p 8000:8000 expert-immo:latest
```

---
*Réalisé dans le cadre de la formation Développeur IA - Promotion 2026.*
