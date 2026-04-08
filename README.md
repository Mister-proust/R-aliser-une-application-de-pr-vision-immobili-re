# Réaliser une application de prévision immobilière

Projet réalisé dans le cadre de la formation Développeur en Intelligence Artificielle avec Cyril J et Melody D

Ce projet a pour but de créer une application web permettant d'estimer la valeur foncière d'un bien immobilier en se basant sur des données historiques. Le projet intègre un pipeline de traitement de données, un modèle de machine learning, et une API pour l'inférence.

## Installation et Lancement

### Installer l'environnement virtuel et les dépendances

```bash
pip install uv #si python est installé, et installation via python
#sinon installation globale
sudo apt install astral-uv
#puis synchronisation de l'environnement virtuel
uv sync
```

### Préparer les données et le modèle avec DVC

Assurez-vous d'avoir les deux fichiers de données originaux dans le dossier `data`: `ValeursFoncieres-2025-S1.txt` et `communes-france-2025.csv`.

Pour préparer les données et entraîner le modèle, exécutez la commande suivante, qui suivra les étapes définies dans `dvc.yaml`:
```bash
uv run dvc repro
```

### Lancer l'application FastAPI

```bash
uv run src/app/main.py
```

Accédez à l'interface web à l'adresse [http://127.0.0.1:8000/](http://127.0.0.1:8000/) et à la documentation interactive de l'API (Swagger UI) à [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/).

### Lancer l'application Gradio (Agent Immobilier Expert)

L'application Gradio propose un agent conversationnel pour estimer la valeur de biens immobiliers.

#### Configurer les variables d'environnement

Copiez le fichier `template.env` en `.env` et renseignez vos clés :

```bash
cp template.env .env
```

Éditez le fichier `.env` avec au minimum votre clé API Mistral :

```
MISTRAL_API_KEY=<votre_clé_api_mistral>
```

Variables optionnelles pour le suivi LangSmith :

```
LANGSMITH_API_KEY=<votre_clé_api_langsmith>
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com ou https://api.smith.langchain.com
LANGSMITH_PROJECT=<nom_de_votre_projet>
```
et pour la gestion des ports:
```
MCP_SERVER_PORT=8001
FASTAPI_PORT=8000
GRADIO_PORT=7860
```
#### Démarrer le serveur MCP

```bash
uv run src/mcp_server/server.py
```

#### Démarrer l'interface Gradio

```bash
uv run src/agentia/main.py
```

Accédez à l'interface de l'agent à l'adresse [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

#### Démarrer l'application web FastAPI

```bash
uv run src/app/main.py
```

---

## Documentation Technique

### Arborescence du Projet

```
.
├── data/                # Données brutes et modèles versionnés par DVC
├── notebooks/           # Espace pour l'exploration et l'analyse de données
├── scripts/             # Scripts pour le nettoyage des données et l'entraînement du modèle
│   ├── clean_dvf.py
│   └── model.py
├── src/app/             # Code source de l'application FastAPI
│   ├── main.py          # Point d'entrée de l'application
│   └── routers/
│       └── endpoints.py # Définition des routes de l'API
├── templates/           # Fichiers HTML pour l'interface utilisateur
├── tests/               # Tests unitaires
├── Dockerfile           # Fichier pour la conteneurisation de l'application
├── dvc.yaml             # Définition du pipeline DVC
└── requirements.txt     # Dépendances Python
```

### Gestion des Données avec DVC

[DVC (Data Version Control)](https://dvc.org/) est utilisé pour versionner les données et les modèles sans les stocker directement dans Git. Le fichier `dvc.yaml` définit les étapes du pipeline :
1.  **`clean_data`**: Nettoyage et préparation du fichier de valeurs foncières (`ValeursFoncieres-2025-S1.txt`) à l'aide de `scripts/clean_dvf.py`.
2.  **`train_model`**: Entraînement d'un modèle de régression (par exemple, LightGBM) sur les données nettoyées à l'aide de `scripts/model.py`. Le modèle entraîné est ensuite sauvegardé dans `data/models/`.

Les fichiers `.dvc` dans le dossier `data/` permettent à DVC de suivre les versions des fichiers de données volumineux.

### Modèle de Machine Learning

Le cœur du projet est un modèle prédictif qui estime la **valeur foncière** d'un bien.
-   **Script d'entraînement**: `scripts/model.py`
-   **Objectif**: Prédire la variable `Valeur fonciere`.
-   **Features utilisées**: Le modèle s'appuie sur des caractéristiques telles que la `Surface reelle bati`, le `Nombre pieces principales`, le type de voie, et des données géographiques issues du fichier des communes.

### Application Web (API)

L'application est construite avec **FastAPI**, un framework web moderne pour Python. Elle expose plusieurs endpoints pour interagir avec le modèle de prédiction.

-   **Interface Utilisateur**: `http://127.0.0.1:8000/`
    -   Une page web simple pour tester le modèle de prédiction de manière interactive.

-   **Documentation de l'API (Swagger UI)**: `http://127.0.0.1:8000/docs`
    -   Documentation interactive de tous les endpoints de l'API, permettant de les tester directement depuis le navigateur.

-   **Endpoint de Prédiction**: `POST /predict`
    -   Ce endpoint attend une requête `POST` avec un corps JSON contenant les caractéristiques du bien immobilier (par exemple, `Surface reelle bati`, `Nombre pieces principales`, etc.).
    -   Il retourne une estimation de la `Valeur fonciere` sous forme de réponse JSON.

### Conteneurisation

Un `Dockerfile` est inclus pour permettre de construire une image Docker de l'application, facilitant ainsi son déploiement dans un environnement conteneurisé.

    #### Construction de l'image Docker

```bash
docker build -t mon-app:latest .
```

#### Démarrage du conteneur

```bash
docker run -p 8000:8000 mon-app:latest
```

### Une VM a été créée sur le datalab via terraform

cf fichier main.tf

### Connexion a la VM.
```bash
ssh group3@ssh3.datalab.centreia.fr
```

pour accéder au logs du container sur la vp, à fins de debug : 
```bash
docker logs group3-app-container
```

### On peut cloner le github du projet
Sur le serevur distant :
```bash
git clone https://github.com/Mister-proust/R-aliser-une-application-de-pr-vision-immobili-re.git
docker build -t g3appimmo .
docker run -p 5003:8000 g3appimmo
```

