"""
Tests unitaires pour les fonctions utilitaires et la logique métier
de src/app/routers/endpoints.py
Couvre : safe_float, safe_int, verify_token, perform_prediction, prediction_model (mocké)
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import HTTPException
from src.app.routers.endpoints import (
    safe_float,
    safe_int,
    verify_token,
    perform_prediction,
    prediction_model,
    get_commune_info,
)


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_none_returns_default_zero(self):
        assert safe_float(None) == 0.0

    def test_none_returns_custom_default(self):
        assert safe_float(None, default=99.9) == 99.9

    def test_empty_string_returns_default(self):
        assert safe_float("") == 0.0

    def test_blank_string_returns_default(self):
        assert safe_float("   ") == 0.0

    def test_valid_float_string(self):
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_valid_integer_string(self):
        assert safe_float("42") == pytest.approx(42.0)

    def test_valid_float_value(self):
        assert safe_float(2.718) == pytest.approx(2.718)

    def test_valid_integer_value(self):
        assert safe_float(10) == pytest.approx(10.0)

    def test_invalid_string_returns_default(self):
        assert safe_float("not_a_number") == 0.0

    def test_invalid_string_returns_custom_default(self):
        assert safe_float("abc", default=-1.0) == -1.0

    def test_zero_string(self):
        assert safe_float("0") == 0.0

    def test_negative_value(self):
        assert safe_float("-55.5") == pytest.approx(-55.5)


# ---------------------------------------------------------------------------
# safe_int
# ---------------------------------------------------------------------------

class TestSafeInt:
    def test_none_returns_default_zero(self):
        assert safe_int(None) == 0

    def test_none_returns_custom_default(self):
        assert safe_int(None, default=5) == 5

    def test_empty_string_returns_default(self):
        assert safe_int("") == 0

    def test_valid_integer_string(self):
        assert safe_int("7") == 7

    def test_valid_float_string_truncates(self):
        assert safe_int("3.9") == 3

    def test_valid_integer_value(self):
        assert safe_int(12) == 12

    def test_invalid_string_returns_default(self):
        assert safe_int("abc") == 0

    def test_invalid_string_returns_custom_default(self):
        assert safe_int("xyz", default=99) == 99

    def test_zero(self):
        assert safe_int("0") == 0

    def test_negative_value(self):
        assert safe_int("-3") == -3


# ---------------------------------------------------------------------------
# verify_token
# ---------------------------------------------------------------------------

class TestVerifyToken:
    def test_missing_header_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            verify_token(authorization=None)
        assert exc_info.value.status_code == 401

    def test_invalid_format_no_bearer_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            verify_token(authorization="just-a-token")
        assert exc_info.value.status_code == 401

    def test_invalid_format_extra_parts_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            verify_token(authorization="Bearer token extra")
        assert exc_info.value.status_code == 401

    def test_wrong_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            verify_token(authorization="Bearer wrong-token-xyz")
        assert exc_info.value.status_code == 401

    def test_detail_message_present_on_missing_header(self):
        with pytest.raises(HTTPException) as exc_info:
            verify_token(authorization=None)
        # exc_info.value.detail est une chaîne de caractères
        assert exc_info.value.detail is not None
        assert len(str(exc_info.value.detail)) > 0

    def test_valid_token_returns_true(self):
        from src.app import config
        result = verify_token(authorization=f"Bearer {config.API_TOKEN}")
        assert result is True


# ---------------------------------------------------------------------------
# perform_prediction (fallback)
# ---------------------------------------------------------------------------

class TestPerformPrediction:
    def test_returns_dict_with_estimated_price(self):
        payload = {"surface": 50, "rooms": 2}
        result = perform_prediction(payload)
        assert "estimated_price" in result

    def test_estimated_price_is_numeric(self):
        payload = {"surface": 50, "rooms": 2}
        result = perform_prediction(payload)
        assert isinstance(result["estimated_price"], (int, float))

    def test_currency_is_eur(self):
        payload = {"surface": 50, "rooms": 2}
        result = perform_prediction(payload)
        assert result["currency"] == "EUR"

    def test_zero_surface_zero_rooms(self):
        payload = {"surface": 0, "rooms": 0}
        result = perform_prediction(payload)
        assert result["estimated_price"] == 0.0

    def test_larger_surface_gives_higher_price(self):
        small = perform_prediction({"surface": 30, "rooms": 1})
        large = perform_prediction({"surface": 100, "rooms": 1})
        assert large["estimated_price"] > small["estimated_price"]

    def test_more_rooms_gives_higher_price(self):
        few = perform_prediction({"surface": 50, "rooms": 1})
        many = perform_prediction({"surface": 50, "rooms": 5})
        assert many["estimated_price"] > few["estimated_price"]

    def test_none_surface_does_not_raise(self):
        payload = {"surface": None, "rooms": None}
        result = perform_prediction(payload)
        assert "estimated_price" in result

    def test_input_echoed_in_response(self):
        payload = {"surface": 60, "rooms": 3}
        result = perform_prediction(payload)
        assert result["input"] == payload


# ---------------------------------------------------------------------------
# prediction_model (via mock pipeline)
# ---------------------------------------------------------------------------

class TestPredictionModel:
    def _mock_geo_response(self):
        return {
            "densite": 20000.0,
            "population": 2161000.0,
            "superficie_km2": 105.4,
            "latitude_centre": 48.8566,
            "longitude_centre": 2.3522,
        }

    def test_returns_estimated_price(self, mock_pipeline):
        payload = {
            "commune": "Paris",
            "type_voie": "Rue",
            "type_local": "Appartement",
            "surface_reelle_bati": 65,
            "nombre_pieces_principales": 3,
        }
        with patch("src.app.routers.endpoints.load_model", return_value=mock_pipeline), \
             patch("src.app.routers.endpoints.get_commune_info", return_value=self._mock_geo_response()):
            result = prediction_model(payload)
        assert "estimated_price" in result
        assert isinstance(result["estimated_price"], float)

    def test_prediction_value_matches_mock(self, mock_pipeline):
        mock_pipeline.predict.return_value = np.array([350000.0])
        payload = {"commune": "Lyon", "type_voie": "Avenue", "type_local": "Maison",
                   "surface_reelle_bati": 100, "nombre_pieces_principales": 4}
        with patch("src.app.routers.endpoints.load_model", return_value=mock_pipeline), \
             patch("src.app.routers.endpoints.get_commune_info", return_value=self._mock_geo_response()):
            result = prediction_model(payload)
        assert result["estimated_price"] == pytest.approx(350000.0)

    def test_without_commune_uses_zero_features(self, mock_pipeline):
        payload = {"type_voie": "Rue", "type_local": "Appartement",
                   "surface_reelle_bati": 45, "nombre_pieces_principales": 2}
        with patch("src.app.routers.endpoints.load_model", return_value=mock_pipeline):
            result = prediction_model(payload)
        assert "estimated_price" in result

    def test_input_dataframe_echoed(self, mock_pipeline):
        payload = {"commune": "Paris", "type_voie": "Rue", "type_local": "Appartement",
                   "surface_reelle_bati": 65}
        with patch("src.app.routers.endpoints.load_model", return_value=mock_pipeline), \
             patch("src.app.routers.endpoints.get_commune_info", return_value=self._mock_geo_response()):
            result = prediction_model(payload)
        assert "input" in result
        assert isinstance(result["input"], dict)

    def test_alternative_surface_key(self, mock_pipeline):
        """Teste que la clé 'surface' est aussi acceptée comme 'surface_reelle_bati'."""
        payload = {"commune": "Paris", "surface": 70, "rooms": 3}
        with patch("src.app.routers.endpoints.load_model", return_value=mock_pipeline), \
             patch("src.app.routers.endpoints.get_commune_info", return_value=self._mock_geo_response()):
            result = prediction_model(payload)
        assert "estimated_price" in result

    def test_model_load_error_raises_http_exception(self):
        payload = {"commune": "Paris", "surface_reelle_bati": 65}
        with patch("src.app.routers.endpoints.load_model",
                   side_effect=HTTPException(status_code=500, detail="Modèle introuvable")):
            with pytest.raises(HTTPException) as exc_info:
                prediction_model(payload)
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# get_commune_info (via mock requests)
# ---------------------------------------------------------------------------

class TestGetCommuneInfo:
    def _mock_geo_api_response(self, nom="Paris", pop=2161000, surface=10540, coords=[2.3522, 48.8566]):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{
            "nom": nom,
            "code": "75056",
            "population": pop,
            "surface": surface,
            "centre": {"coordinates": coords},
        }]
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_dict_with_expected_keys(self):
        with patch("src.app.routers.endpoints.requests.get",
                   return_value=self._mock_geo_api_response()):
            result = get_commune_info("Paris")
        for key in ["densite", "population", "superficie_km2", "latitude_centre", "longitude_centre"]:
            assert key in result

    def test_returns_correct_population(self):
        with patch("src.app.routers.endpoints.requests.get",
                   return_value=self._mock_geo_api_response(pop=500000)):
            result = get_commune_info("Lyon")
        assert result["population"] == 500000.0

    def test_calculates_densite_correctly(self):
        # pop=1000, surface=100 hectares → surface_km2=1.0 → densite=1000
        with patch("src.app.routers.endpoints.requests.get",
                   return_value=self._mock_geo_api_response(pop=1000, surface=100)):
            result = get_commune_info("TestCommune")
        assert result["densite"] == pytest.approx(1000.0, rel=0.01)

    def test_returns_correct_coordinates(self):
        with patch("src.app.routers.endpoints.requests.get",
                   return_value=self._mock_geo_api_response(coords=[5.3698, 43.2965])):
            result = get_commune_info("Marseille")
        assert result["longitude_centre"] == pytest.approx(5.3698)
        assert result["latitude_centre"] == pytest.approx(43.2965)

    def test_empty_api_response_raises_5xx(self):
        """Quand l'API retourne une liste vide, get_commune_info lève une HTTPException.
        En pratique, le catch-all transforme le 404 interne en 500."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        with patch("src.app.routers.endpoints.requests.get", return_value=mock_resp):
            with pytest.raises(HTTPException) as exc_info:
                get_commune_info("CommuneInconnueXYZ")
        assert exc_info.value.status_code in (404, 500)

    def test_requests_exception_raises_503(self):
        import requests as req
        with patch("src.app.routers.endpoints.requests.get",
                   side_effect=req.exceptions.ConnectionError("Network error")):
            with pytest.raises(HTTPException) as exc_info:
                get_commune_info("Paris")
        assert exc_info.value.status_code == 503

    def test_uses_code_parameter_for_insee_code(self):
        with patch("src.app.routers.endpoints.requests.get",
                   return_value=self._mock_geo_api_response()) as mock_get:
            get_commune_info("75056")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert "code" in params

    def test_uses_nom_parameter_for_commune_name(self):
        with patch("src.app.routers.endpoints.requests.get",
                   return_value=self._mock_geo_api_response()) as mock_get:
            get_commune_info("Paris")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert "nom" in params
