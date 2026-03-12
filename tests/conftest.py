import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Fixtures globales
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fastapi_app():
    """Crée l'application FastAPI (chargée une seule fois par session)."""
    from src.app.main import app as application
    return application


@pytest.fixture
def client(fastapi_app):
    """Client de test HTTP pour l'application FastAPI."""
    from fastapi.testclient import TestClient
    return TestClient(fastapi_app)


@pytest.fixture
def valid_token():
    """Retourne le token API valide depuis la configuration."""
    from src.app import config
    return config.API_TOKEN


@pytest.fixture
def auth_headers(valid_token):
    """Headers HTTP avec un token valide pour les routes protégées."""
    return {"Authorization": f"Bearer {valid_token}"}


@pytest.fixture
def bad_auth_headers():
    """Headers HTTP avec un token invalide."""
    return {"Authorization": "Bearer mauvais-token-inconnu"}


# ---------------------------------------------------------------------------
# DataFrames de test
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dvf_df():
    """DataFrame DVF minimal représentatif pour les tests unitaires."""
    return pd.DataFrame({
        "Date mutation": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05"]),
        "Nature mutation": ["Vente", "Vente", "Echange", "Vente"],
        "Valeur fonciere": pd.array([250000.0, None, 150000.0, 300000.0], dtype="float32"),
        "No voie": ["12", "5", "8", "1"],
        "B/T/Q": [None, "B", None, None],
        "Type de voie": ["RUE", "AV", "BD:", "IMP"],
        "Code voie": ["0001", "0002", "0003", "0004"],
        "Voie": ["RUE DE LA PAIX", "AVENUE DE PARIS", "BOULEVARD VICTOR", "IMPASSE DES FLEURS"],
        "Code postal": ["75001", "75002", "75003", "75004"],
        "Commune": ["PARIS", "PARIS", "PARIS", "PARIS"],
        "Code departement": ["75", "75", "75", "75"],
        "Code commune": ["056", "056", "056", "056"],
        "Prefixe de section": [None, None, None, None],
        "Section": ["AB", "CD", "EF", "GH"],
        "No plan": pd.array([1, 2, 3, 4], dtype="Int16"),
        "Code type local": pd.array([2, 1, 3, 2], dtype="Int8"),
        "Type local": ["Appartement", "Maison", "Appartement", "Appartement"],
        "Surface reelle bati": pd.array([65, 120, None, 45], dtype="Int32"),
        "Nombre pieces principales": pd.array([3, 5, None, 2], dtype="Int8"),
        "Surface terrain": pd.array([None, 500, None, None], dtype="Int32"),
    })


@pytest.fixture
def sample_communes_df():
    """DataFrame communes minimal pour les tests unitaires."""
    return pd.DataFrame({
        "code_insee": ["75056", "69123", "13055"],
        "nom_standard": ["Paris", "Lyon", "Marseille"],
        "nom_standard_majuscule": ["PARIS", "LYON", "MARSEILLE"],
        "population": [2161000, 522000, 873000],
        "superficie_km2": [105.4, 47.9, 240.6],
        "densite": [20494.3, 10897.7, 3630.9],
        "latitude_centre": [48.8566, 45.7640, 43.2965],
        "longitude_centre": [2.3522, 4.8357, 5.3698],
    })


@pytest.fixture
def mock_pipeline():
    """Pipeline ML mocké pour éviter de charger le vrai fichier .pkl."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([250000.0])
    pipeline.named_steps = {
        "pre": MagicMock(),
        "model": MagicMock(),
    }
    return pipeline
