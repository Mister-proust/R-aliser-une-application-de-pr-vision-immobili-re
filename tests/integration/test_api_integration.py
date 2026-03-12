"""
Tests d'intégration pour l'API FastAPI (src/app)
Couvre : tous les endpoints, enchaînement des appels, cohérence des réponses
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from src.app.main import app


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def api_token():
    from src.app import config
    return config.API_TOKEN


@pytest.fixture(scope="module")
def auth_headers(api_token):
    return {"Authorization": f"Bearer {api_token}"}


def _mock_pipeline(price=250000.0):
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([price])
    pipeline.named_steps = {"pre": MagicMock(), "model": MagicMock()}
    return pipeline


def _mock_commune_features():
    return {
        "densite": 20000.0,
        "population": 2161000.0,
        "superficie_km2": 105.4,
        "latitude_centre": 48.8566,
        "longitude_centre": 2.3522,
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self, client):
        r = client.get("/health")
        assert r.json() == {"status": "ok"}

    def test_health_is_json(self, client):
        r = client.get("/health")
        assert "application/json" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# /meta/options
# ---------------------------------------------------------------------------

class TestMetaOptionsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/meta/options")
        assert r.status_code == 200

    def test_returns_type_voie_list(self, client):
        r = client.get("/meta/options")
        data = r.json()
        assert "type_voie" in data
        assert isinstance(data["type_voie"], list)
        assert len(data["type_voie"]) > 0

    def test_returns_type_local_list(self, client):
        r = client.get("/meta/options")
        data = r.json()
        assert "type_local" in data
        assert isinstance(data["type_local"], list)
        assert len(data["type_local"]) > 0

    def test_type_voie_contains_rue(self, client):
        r = client.get("/meta/options")
        assert "Rue" in r.json()["type_voie"]

    def test_type_local_contains_appartement(self, client):
        r = client.get("/meta/options")
        assert "Appartement" in r.json()["type_local"]

    def test_type_local_contains_maison(self, client):
        r = client.get("/meta/options")
        assert "Maison" in r.json()["type_local"]


# ---------------------------------------------------------------------------
# GET / (home page)
# ---------------------------------------------------------------------------

class TestHomePageEndpoint:
    def test_home_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_home_returns_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers.get("content-type", "")

    def test_home_body_not_empty(self, client):
        r = client.get("/")
        assert len(r.content) > 0


# ---------------------------------------------------------------------------
# GET /prediction (page formulaire)
# ---------------------------------------------------------------------------

class TestPredictionPageEndpoint:
    def test_prediction_page_returns_200(self, client):
        r = client.get("/prediction")
        assert r.status_code == 200

    def test_prediction_page_returns_html(self, client):
        r = client.get("/prediction")
        assert "text/html" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# POST /predict — Authentification
# ---------------------------------------------------------------------------

class TestPredictAuthentication:
    def test_no_token_returns_401(self, client):
        r = client.post("/predict", json={"surface_reelle_bati": 65, "type_local": "Appartement"})
        assert r.status_code == 401

    def test_wrong_token_returns_401(self, client):
        headers = {"Authorization": "Bearer mauvais-token-xyz"}
        r = client.post("/predict", json={"surface_reelle_bati": 65}, headers=headers)
        assert r.status_code == 401

    def test_malformed_header_returns_401(self, client):
        headers = {"Authorization": "just-a-string-no-bearer"}
        r = client.post("/predict", json={}, headers=headers)
        assert r.status_code == 401

    def test_missing_bearer_prefix_returns_401(self, client, api_token):
        headers = {"Authorization": api_token}  # sans "Bearer "
        r = client.post("/predict", json={}, headers=headers)
        assert r.status_code == 401

    def test_401_response_has_detail(self, client):
        r = client.post("/predict", json={})
        assert "detail" in r.json()


# ---------------------------------------------------------------------------
# POST /predict — Prédiction
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_valid_token_returns_200(self, client, auth_headers):
        with patch("app.routers.endpoints.load_model", return_value=_mock_pipeline()), \
             patch("app.routers.endpoints.get_commune_info", return_value=_mock_commune_features()):
            r = client.post("/predict",
                            json={"commune": "Paris", "type_voie": "Rue",
                                  "type_local": "Appartement", "surface_reelle_bati": 65,
                                  "nombre_pieces_principales": 3},
                            headers=auth_headers)
        assert r.status_code == 200

    def test_response_contains_estimated_price(self, client, auth_headers):
        with patch("app.routers.endpoints.load_model", return_value=_mock_pipeline(280000.0)), \
             patch("app.routers.endpoints.get_commune_info", return_value=_mock_commune_features()):
            r = client.post("/predict",
                            json={"commune": "Paris", "type_voie": "Rue",
                                  "type_local": "Appartement", "surface_reelle_bati": 65},
                            headers=auth_headers)
        data = r.json()
        assert "estimated_price" in data or ("result" in data and "estimated_price" in data["result"])

    def test_estimated_price_is_numeric(self, client, auth_headers):
        with patch("app.routers.endpoints.load_model", return_value=_mock_pipeline(250000.0)), \
             patch("app.routers.endpoints.get_commune_info", return_value=_mock_commune_features()):
            r = client.post("/predict",
                            json={"commune": "Lyon", "type_local": "Maison", "surface_reelle_bati": 100},
                            headers=auth_headers)
        data = r.json()
        price = data.get("estimated_price") or data.get("result", {}).get("estimated_price")
        assert isinstance(price, (int, float))

    def test_fallback_when_model_unavailable(self, client, auth_headers):
        """Quand prediction_model échoue, le fallback doit être retourné."""
        with patch("app.routers.endpoints.prediction_model",
                   side_effect=Exception("Modèle absent")):
            r = client.post("/predict",
                            json={"surface": 60, "rooms": 3},
                            headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "warning" in data or "result" in data

    def test_empty_payload_returns_200(self, client, auth_headers):
        """Un payload vide ne doit pas faire crasher l'API (fallback attendu)."""
        with patch("app.routers.endpoints.prediction_model",
                   side_effect=Exception("Modèle absent")):
            r = client.post("/predict", json={}, headers=auth_headers)
        assert r.status_code == 200

    def test_commune_not_found_returns_404(self, client, auth_headers):
        """Si la commune n'existe pas, prediction_model lève une 404."""
        from fastapi import HTTPException
        with patch("app.routers.endpoints.prediction_model",
                   side_effect=HTTPException(status_code=404, detail="Commune non trouvée: 'XYZ'")):
            r = client.post("/predict",
                            json={"commune": "XYZ", "surface_reelle_bati": 65},
                            headers=auth_headers)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /shap-explanation — Authentification
# ---------------------------------------------------------------------------

class TestShapAuthentication:
    def test_no_token_returns_401(self, client):
        r = client.post("/shap-explanation", json={"surface_reelle_bati": 65})
        assert r.status_code == 401

    def test_wrong_token_returns_401(self, client):
        headers = {"Authorization": "Bearer fake-token"}
        r = client.post("/shap-explanation", json={"surface_reelle_bati": 65}, headers=headers)
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# GET /map-data
# ---------------------------------------------------------------------------

class TestMapDataEndpoint:
    def test_missing_params_returns_422(self, client):
        """Sans les query params requis, FastAPI doit retourner 422."""
        r = client.get("/map-data")
        assert r.status_code == 422

    def test_with_all_params_returns_valid_response(self, client):
        """Avec tous les paramètres requis, l'endpoint répond (200 si DB présente, 500 sinon)."""
        r = client.get("/map-data?south=48.0&west=2.0&north=49.0&east=3.0&zoom=10")
        assert r.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Enchaînement multi-endpoints (flux complet)
# ---------------------------------------------------------------------------

class TestApiFlow:
    def test_health_then_meta_then_predict(self, client, auth_headers):
        """Vérifie un enchaînement réaliste d'appels."""
        # 1. Vérifier la santé
        health = client.get("/health")
        assert health.status_code == 200

        # 2. Récupérer les options
        options = client.get("/meta/options")
        assert options.status_code == 200
        type_local = options.json()["type_local"][0]

        # 3. Faire une prédiction avec un type récupéré dynamiquement
        # On patche prediction_model pour retourner directement le résultat
        mock_result = {"estimated_price": 200000.0, "currency": "EUR", "input": {}}
        with patch("app.routers.endpoints.prediction_model", return_value=mock_result):
            predict = client.post("/predict",
                                  json={"commune": "Paris", "type_local": type_local,
                                        "surface_reelle_bati": 55, "nombre_pieces_principales": 2},
                                  headers=auth_headers)
        assert predict.status_code == 200
