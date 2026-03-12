"""
Tests système pour l'application complète
Simule des scénarios utilisateur de bout en bout via l'API REST.
Le modèle ML et l'API Géo externe sont mockés pour garantir
la reproductibilité sans dépendances réseau ou fichiers lourds.
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from src.app.main import app
from src.app import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_headers():
    return {"Authorization": f"Bearer {config.API_TOKEN}"}


def _pipeline(price=250000.0):
    p = MagicMock()
    p.predict.return_value = np.array([price])
    p.named_steps = {"pre": MagicMock(), "model": MagicMock()}
    return p


def _commune_features(city="Paris", densite=20000.0, lat=48.8566, lon=2.3522):
    return {
        "densite": densite,
        "population": 2161000.0,
        "superficie_km2": 105.4,
        "latitude_centre": lat,
        "longitude_centre": lon,
    }


# ---------------------------------------------------------------------------
# Scénario 1 — Utilisateur découvre et utilise l'API pour la première fois
# ---------------------------------------------------------------------------

class TestScenarioDecouvertAPI:
    def test_api_est_accessible(self, client):
        """L'API répond immédiatement sur /health."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_docs_swagger_accessible(self, client):
        """La documentation Swagger est disponible."""
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_accessible(self, client):
        """La documentation ReDoc est disponible."""
        r = client.get("/redoc")
        assert r.status_code == 200

    def test_formulaire_accessible(self, client):
        """La page de formulaire HTML est accessible."""
        r = client.get("/prediction")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_page_accueil_accessible(self, client):
        """La page d'accueil est accessible."""
        r = client.get("/")
        assert r.status_code == 200

    def test_options_disponibles_avant_prediction(self, client):
        """L'utilisateur peut récupérer les options avant de soumettre le formulaire."""
        r = client.get("/meta/options")
        assert r.status_code == 200
        data = r.json()
        assert "type_voie" in data
        assert "type_local" in data


# ---------------------------------------------------------------------------
# Scénario 2 — Estimation d'un appartement parisien
# ---------------------------------------------------------------------------

class TestScenarioAppartementParis:
    PAYLOAD = {
        "commune": "Paris",
        "type_voie": "Rue",
        "type_local": "Appartement",
        "surface_reelle_bati": 65,
        "nombre_pieces_principales": 3,
        "surface_terrain": 0,
    }

    def test_estimation_sans_token_refusee(self, client):
        r = client.post("/predict", json=self.PAYLOAD)
        assert r.status_code == 401

    def test_estimation_avec_mauvais_token_refusee(self, client):
        headers = {"Authorization": "Bearer faux-token-pirate"}
        r = client.post("/predict", json=self.PAYLOAD, headers=headers)
        assert r.status_code == 401

    def test_estimation_avec_token_valide_retourne_prix(self, client, auth_headers):
        mock_result = {"estimated_price": 320000.0, "currency": "EUR", "input": {}}
        with patch("app.routers.endpoints.prediction_model", return_value=mock_result):
            r = client.post("/predict", json=self.PAYLOAD, headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        price = data.get("estimated_price") or data.get("result", {}).get("estimated_price")
        assert isinstance(price, (int, float))
        assert price > 0

    def test_prix_correspond_a_prediction_modele(self, client, auth_headers):
        expected_price = 320000.0
        mock_result = {"estimated_price": expected_price, "currency": "EUR", "input": {}}
        with patch("app.routers.endpoints.prediction_model", return_value=mock_result):
            r = client.post("/predict", json=self.PAYLOAD, headers=auth_headers)
        data = r.json()
        price = data.get("estimated_price") or data.get("result", {}).get("estimated_price")
        assert price == pytest.approx(expected_price, rel=0.01)

    def test_reponse_contient_devise_eur(self, client, auth_headers):
        mock_result = {"estimated_price": 320000.0, "currency": "EUR", "input": {}}
        with patch("app.routers.endpoints.prediction_model", return_value=mock_result):
            r = client.post("/predict", json=self.PAYLOAD, headers=auth_headers)
        data = r.json()
        currency = data.get("currency") or data.get("result", {}).get("currency")
        assert currency == "EUR"


# ---------------------------------------------------------------------------
# Scénario 3 — Estimation d'une maison en province
# ---------------------------------------------------------------------------

class TestScenarioMaisonProvince:
    PAYLOAD_MAISON = {
        "commune": "Lyon",
        "type_voie": "Avenue",
        "type_local": "Maison",
        "surface_reelle_bati": 130,
        "nombre_pieces_principales": 5,
        "surface_terrain": 400,
    }

    def test_maison_retourne_estimation(self, client, auth_headers):
        mock_result = {"estimated_price": 450000.0, "currency": "EUR", "input": {}}
        with patch("app.routers.endpoints.prediction_model", return_value=mock_result):
            r = client.post("/predict", json=self.PAYLOAD_MAISON, headers=auth_headers)
        assert r.status_code == 200

    def test_maison_plus_grande_peut_etre_plus_chere_que_appartement(self, client, auth_headers):
        """Vérifie la cohérence logique : maison > appartement petit (selon le mock)."""
        with patch("app.routers.endpoints.prediction_model",
                   return_value={"estimated_price": 450000.0, "currency": "EUR", "input": {}}):
            r_maison = client.post("/predict", json=self.PAYLOAD_MAISON, headers=auth_headers)
        with patch("app.routers.endpoints.prediction_model",
                   return_value={"estimated_price": 200000.0, "currency": "EUR", "input": {}}):
            r_appart = client.post("/predict",
                                   json={**self.PAYLOAD_MAISON, "surface_reelle_bati": 35,
                                         "type_local": "Appartement"},
                                   headers=auth_headers)
        price_maison = r_maison.json().get("estimated_price", 0)
        price_appart = r_appart.json().get("estimated_price", 0)
        assert price_maison > price_appart


# ---------------------------------------------------------------------------
# Scénario 4 — Robustesse aux données invalides ou manquantes
# ---------------------------------------------------------------------------

class TestScenarioRobustesse:
    def test_payload_vide_ne_plante_pas(self, client, auth_headers):
        with patch("app.routers.endpoints.prediction_model",
                   side_effect=Exception("Modèle absent")):
            r = client.post("/predict", json={}, headers=auth_headers)
        assert r.status_code == 200  # fallback attendu

    def test_commune_inconnue_retourne_404(self, client, auth_headers):
        from fastapi import HTTPException
        with patch("app.routers.endpoints.prediction_model",
                   side_effect=HTTPException(status_code=404, detail="Commune non trouvée")):
            r = client.post("/predict",
                            json={"commune": "VilleQuiNexistePas", "surface_reelle_bati": 60},
                            headers=auth_headers)
        assert r.status_code == 404

    def test_surface_nulle_utilisee_dans_fallback(self, client, auth_headers):
        """Surface 0 + rooms 0 via la formule fallback doit retourner estimated_price = 0."""
        with patch("app.routers.endpoints.prediction_model",
                   side_effect=Exception("Modèle absent")):
            r = client.post("/predict",
                            json={"surface": 0, "rooms": 0},
                            headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        price = data.get("estimated_price") or data.get("result", {}).get("estimated_price")
        assert price == pytest.approx(0.0)

    def test_appels_consecutifs_coherents(self, client, auth_headers):
        """Plusieurs appels avec le même payload doivent retourner le même prix."""
        payload = {"commune": "Paris", "type_voie": "Rue",
                   "type_local": "Appartement", "surface_reelle_bati": 70}
        mock_result = {"estimated_price": 300000.0, "currency": "EUR", "input": {}}
        with patch("app.routers.endpoints.prediction_model", return_value=mock_result):
            res1 = client.post("/predict", json=payload, headers=auth_headers).json()
            res2 = client.post("/predict", json=payload, headers=auth_headers).json()
        price1 = res1.get("estimated_price") or res1.get("result", {}).get("estimated_price")
        price2 = res2.get("estimated_price") or res2.get("result", {}).get("estimated_price")
        assert price1 == price2


# ---------------------------------------------------------------------------
# Scénario 5 — Intégrité de l'API (headers, CORS, contenu)
# ---------------------------------------------------------------------------

class TestScenarioSecurite:
    def test_api_retourne_json_sur_health(self, client):
        r = client.get("/health")
        assert "application/json" in r.headers.get("content-type", "")

    def test_reponse_401_contient_detail(self, client):
        r = client.post("/predict", json={})
        assert "detail" in r.json()

    def test_reponse_422_sur_map_data_sans_params(self, client):
        """Les paramètres requis de /map-data sont validés par FastAPI."""
        r = client.get("/map-data")
        assert r.status_code == 422
        assert "detail" in r.json()

    def test_health_endpoint_ne_requiert_pas_de_token(self, client):
        """Le health check est public."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_meta_options_ne_requiert_pas_de_token(self, client):
        """Les options sont publiques."""
        r = client.get("/meta/options")
        assert r.status_code == 200
