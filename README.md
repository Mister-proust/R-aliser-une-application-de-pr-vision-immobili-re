# Réaliser-une-application-de-prévision-immobilière
Projet réalisé dans le cadre de la formation Développeur en Intelligence Artificielle avec Cyril J et Melody D

### Installer l'environnement virtuel et les dépendances

```bash
python -m venv venv
venv\Scripts\activate  # Sur Windows (source venv/bin/activate sur macOS/Linux)
pip install -r requirements.txt
```

### Préparer les données et le modèle avec DVC

S'assurer d'avoir les deux fichiers de données originaux dans le dossier data: `ValeursFoncieres-2025-S1.txt` et `communes-france-2025.csv`

Puis pour préparer les données:
```bash
dvc repro
```

### Lancer l'application

```bash
python app.py
```

accéder à l'interface web à l'adresse [http://127.0.0.1:8000/](http://127.0.0.1:8000/), et au swagger UI à [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/).