Template FastAPI en Clean Architecture (OracleDB)

Ce projet est un template de démarrage pour une application FastAPI, conçu en suivant les principes de la Clean Architecture. Il est spécifiquement documenté pour les développeurs qui découvrent ce patron d'architecture.

Il utilise oracledb pour des requêtes SQL pures, python-jose pour l'authentification JWT, et Pydantic Settings pour la configuration.

Philosophie de la Clean Architecture

L'objectif principal est la séparation des préoccupations. Nous divisons l'application en couches indépendantes :

src/domain (Domaine) : Le cœur de l'application. Contient les modèles métier (Entités). Il ne dépend d'aucune autre couche.

src/application (Application) : Contient la logique métier (Cas d'utilisation / Services). Définit des interfaces (contrats) pour les opérations externes (ex: base de données). Il ne dépend que du domain.

src/infrastructure (Infrastructure) : Fournit l'implémentation concrète des interfaces de l'application (ex: connexion à la base de données Oracle, service JWT). C'est le "comment" technique. Il dépend de l'application.

src/api (Présentation/API) : La couche d'exposition (FastAPI). Gère les requêtes HTTP, la sérialisation et la validation. Elle dépend de l'application.

La règle la plus importante est la Règle de Dépendance : les dépendances vont toujours vers l'intérieur (api -> application <- infrastructure). Le domain est au centre et ne connaît personne.

[Image d'un diagramme simple de la Clean Architecture]

1. Installation

Clonez ce dépôt.

Il est fortement recommandé d'utiliser un environnement virtuel :

python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate


Installez les dépendances requises :

pip install -r requirements.txt


2. Configuration

Ce projet utilise un fichier .env pour gérer les secrets.

Créez un fichier .env à la racine du projet en copiant l'exemple :

cp .env.example .env


Ouvrez le fichier .env et remplissez les valeurs. La bibliothèque oracledb n'utilise pas une URL complète, mais des paramètres séparés :

DB_USER : Votre nom d'utilisateur Oracle.

DB_PASSWORD : Votre mot de passe.

DB_DSN : Le "Data Source Name" d'Oracle, généralement sous la forme hostname:port/service_name (ex: localhost:1521/XEPDB1).

JWT_SECRET_KEY : Une chaîne aléatoire longue et secrète (générez-en une).

JWT_ALGORITHM : Laissez HS256.

JWT_EXPIRE_MINUTES : Durée de validité du token (ex: 30).

3. Lancer le serveur (Localement)

Une fois configuré, vous pouvez lancer le serveur avec uvicorn :

# L'option --reload surveille les changements de fichiers
uvicorn src.main:app --reload


Le serveur sera accessible à l'adresse http://127.0.0.1:8000.
La documentation interactive (Swagger UI) sera disponible à http://127.0.0.1:8000/docs.

4. Lancer le serveur (Docker)

Le projet inclut un Dockerfile optimisé (multi-stage) et un docker-compose.yml.

Assurez-vous que Docker et Docker Compose sont installés.

Assurez-vous que votre fichier .env est correctement rempli.

Lancez le service :

docker-compose up --build


L'application sera accessible de la même manière sur http://127.0.0.1:8000.

5. Lancer les Tests

Ce projet utilise pytest. Les tests sont divisés en :

tests/unit : Tests unitaires qui n'ont aucune dépendance externe (la base de données est "mockée").

tests/integration : Tests d'intégration qui vérifient les routes de l'API (en utilisant un TestClient et en "mockant" la base de données).

Pour lancer tous les tests :

pytest
