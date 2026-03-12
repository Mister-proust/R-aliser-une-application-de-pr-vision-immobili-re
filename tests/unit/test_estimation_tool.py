"""
Tests unitaires pour src/agentia/estimation_tool.py
Couvre : safe_float, safe_int, get_commune_info, estimate_property (avec mocks)
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agentia.estimation_tool import (
    safe_float,
    safe_int,
    get_commune_info,
    estimate_property,
)


# ---------------------------------------------------------------------------
# safe_float (version estimation_tool)
# ---------------------------------------------------------------------------

class TestSafeFloatEstimation:
    def test_none_returns_default(self):
        assert safe_float(None) == 0.0

    def test_empty_string_returns_default(self):
        assert safe_float("") == 0.0

    def test_valid_float_string(self):
        assert safe_float("12.5") == pytest.approx(12.5)

    def test_invalid_string_returns_default(self):
        assert safe_float("abc") == 0.0

    def test_integer_value(self):
        assert safe_float(7) == pytest.approx(7.0)

    def test_custom_default(self):
        assert safe_float(None, default=-1.0) == -1.0


# ---------------------------------------------------------------------------
# safe_int (version estimation_tool)
# ---------------------------------------------------------------------------

class TestSafeIntEstimation:
    def test_none_returns_default(self):
        assert safe_int(None) == 0

    def test_empty_string_returns_default(self):
        assert safe_int("") == 0

    def test_valid_integer_string(self):
        assert safe_int("5") == 5

    def test_float_string_truncates(self):
        assert safe_int("4.9") == 4

    def test_invalid_string_returns_default(self):
        assert safe_int("abc") == 0

    def test_custom_default(self):
        assert safe_int(None, default=10) == 10


# ---------------------------------------------------------------------------
# get_commune_info (version estimation_tool)
# ---------------------------------------------------------------------------

class TestGetCommuneInfoEstimation:
    def _mock_api_response(self, pop=2161000, surface=10540, coords=[2.3522, 48.8566]):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{
            "nom": "Paris",
            "code": "75056",
            "population": pop,
            "surface": surface,
            "centre": {"coordinates": coords},
        }]
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_dict_with_expected_keys(self):
        with patch("src.agentia.estimation_tool.requests.get",
                   return_value=self._mock_api_response()):
            result = get_commune_info("Paris")
        assert result is not None
        for key in ["densite", "population", "superficie_km2", "latitude_centre", "longitude_centre"]:
            assert key in result

    def test_empty_response_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        with patch("src.agentia.estimation_tool.requests.get", return_value=mock_resp):
            result = get_commune_info("CommuneInexistante")
        assert result is None

    def test_exception_returns_none(self):
        import requests as req
        with patch("src.agentia.estimation_tool.requests.get",
                   side_effect=req.exceptions.ConnectionError("Network error")):
            result = get_commune_info("Paris")
        assert result is None

    def test_calculates_densite(self):
        # pop=2000, surface=200 hectares → km2=2.0 → densite=1000
        with patch("src.agentia.estimation_tool.requests.get",
                   return_value=self._mock_api_response(pop=2000, surface=200)):
            result = get_commune_info("TestVille")
        assert result["densite"] == pytest.approx(1000.0, rel=0.01)

    def test_coordinates_extracted_correctly(self):
        with patch("src.agentia.estimation_tool.requests.get",
                   return_value=self._mock_api_response(coords=[5.3698, 43.2965])):
            result = get_commune_info("Marseille")
        assert result["longitude_centre"] == pytest.approx(5.3698)
        assert result["latitude_centre"] == pytest.approx(43.2965)

    def test_zero_surface_gives_zero_densite(self):
        with patch("src.agentia.estimation_tool.requests.get",
                   return_value=self._mock_api_response(pop=1000, surface=0)):
            result = get_commune_info("VilleVide")
        assert result["densite"] == 0.0


# ---------------------------------------------------------------------------
# estimate_property
# ---------------------------------------------------------------------------

class TestEstimateProperty:
    def _mock_commune_info(self):
        return {
            "densite": 20000.0,
            "population": 2161000.0,
            "superficie_km2": 105.4,
            "latitude_centre": 48.8566,
            "longitude_centre": 2.3522,
        }

    def _mock_pipeline(self, price=275000.0):
        pipeline = MagicMock()
        pipeline.predict.return_value = np.array([price])
        return pipeline

    def test_returns_string(self):
        with patch("src.agentia.estimation_tool.load_model", return_value=self._mock_pipeline()), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            result = estimate_property.invoke({
                "commune": "Paris", "type_bien": "Appartement", "surface": 65, "rooms": 3
            })
        assert isinstance(result, str)

    def test_contains_price_in_result(self):
        with patch("src.agentia.estimation_tool.load_model", return_value=self._mock_pipeline(300000.0)), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            result = estimate_property.invoke({
                "commune": "Paris", "type_bien": "Maison", "surface": 100, "rooms": 4
            })
        assert "300" in result  # Le prix doit apparaître formaté

    def test_contains_commune_name(self):
        with patch("src.agentia.estimation_tool.load_model", return_value=self._mock_pipeline()), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            result = estimate_property.invoke({
                "commune": "Lyon", "type_bien": "Appartement", "surface": 50, "rooms": 2
            })
        assert "Lyon" in result

    def test_commune_not_found_returns_error_message(self):
        with patch("src.agentia.estimation_tool.load_model", return_value=self._mock_pipeline()), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=None):
            result = estimate_property.invoke({
                "commune": "VilleInexistante", "type_bien": "Appartement", "surface": 50, "rooms": 2
            })
        assert "VilleInexistante" in result or "Désolé" in result

    def test_model_exception_returns_error_message(self):
        with patch("src.agentia.estimation_tool.load_model",
                   side_effect=Exception("Modèle introuvable")):
            result = estimate_property.invoke({
                "commune": "Paris", "type_bien": "Appartement", "surface": 65, "rooms": 3
            })
        assert "erreur" in result.lower() or "Erreur" in result

    def test_calls_pipeline_predict(self):
        mock_pipeline = self._mock_pipeline()
        with patch("src.agentia.estimation_tool.load_model", return_value=mock_pipeline), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            estimate_property.invoke({
                "commune": "Paris", "type_bien": "Appartement", "surface": 65, "rooms": 3
            })
        mock_pipeline.predict.assert_called_once()

    def test_surface_passed_to_dataframe(self):
        mock_pipeline = self._mock_pipeline()
        with patch("src.agentia.estimation_tool.load_model", return_value=mock_pipeline), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            estimate_property.invoke({
                "commune": "Paris", "type_bien": "Appartement", "surface": 80, "rooms": 3
            })
        call_args = mock_pipeline.predict.call_args[0][0]
        assert call_args["Surface reelle bati"].iloc[0] == pytest.approx(80.0)

    def test_appartement_type_local_mapped(self):
        mock_pipeline = self._mock_pipeline()
        with patch("src.agentia.estimation_tool.load_model", return_value=mock_pipeline), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            estimate_property.invoke({
                "commune": "Paris", "type_bien": "Appartement", "surface": 65, "rooms": 3
            })
        call_args = mock_pipeline.predict.call_args[0][0]
        assert call_args["Type local"].iloc[0] == "Appartement"

    def test_contains_surface_in_result(self):
        with patch("src.agentia.estimation_tool.load_model", return_value=self._mock_pipeline()), \
             patch("src.agentia.estimation_tool.get_commune_info", return_value=self._mock_commune_info()):
            result = estimate_property.invoke({
                "commune": "Paris", "type_bien": "Appartement", "surface": 75, "rooms": 3
            })
        assert "75" in result
