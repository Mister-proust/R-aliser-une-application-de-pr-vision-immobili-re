"""
Tests unitaires pour src/agentia/geocoding_tool.py
Couvre : geocoding_search, reverse_geocoding (avec mock des appels HTTP)
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_server.geocoding_tool import geocoding_search, reverse_geocoding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature(label="73 Avenue de Paris, Paris", lon=2.3522, lat=48.8566, score=0.95):
    return {
        "type": "Feature",
        "properties": {"label": label, "score": score},
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def _mock_response(features=None, raise_for_status=None):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"features": features or []}
    if raise_for_status:
        mock_resp.raise_for_status.side_effect = raise_for_status
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# geocoding_search
# ---------------------------------------------------------------------------

class TestGeocodingSearch:
    def test_returns_string(self):
        mock_resp = _mock_response(features=[_make_feature()])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = geocoding_search.invoke({"q": "73 avenue de Paris"})
        assert isinstance(result, str)

    def test_contains_label_in_result(self):
        label = "73 Avenue de Paris, Paris"
        mock_resp = _mock_response(features=[_make_feature(label=label)])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = geocoding_search.invoke({"q": "73 avenue de Paris"})
        assert label in result

    def test_contains_coordinates_in_result(self):
        mock_resp = _mock_response(features=[_make_feature(lon=2.3522, lat=48.8566)])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = geocoding_search.invoke({"q": "Paris"})
        assert "2.3522" in result
        assert "48.8566" in result

    def test_no_results_returns_aucun_resultat(self):
        mock_resp = _mock_response(features=[])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = geocoding_search.invoke({"q": "adresse_totalement_inexistante_xyz"})
        assert "Aucun résultat" in result

    def test_multiple_results_returns_all(self):
        features = [
            _make_feature("Adresse 1, Paris", lon=2.35, lat=48.85),
            _make_feature("Adresse 2, Lyon", lon=4.83, lat=45.76),
        ]
        mock_resp = _mock_response(features=features)
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = geocoding_search.invoke({"q": "rue"})
        assert "Adresse 1" in result
        assert "Adresse 2" in result

    def test_http_error_returns_error_message(self):
        import requests as req
        with patch("src.agentia.geocoding_tool.requests.get",
                   side_effect=req.exceptions.ConnectionError("Erreur réseau")):
            result = geocoding_search.invoke({"q": "Paris"})
        assert "erreur" in result.lower()

    def test_calls_correct_endpoint(self):
        mock_resp = _mock_response(features=[_make_feature()])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp) as mock_get:
            geocoding_search.invoke({"q": "Paris"})
        called_url = mock_get.call_args[0][0]
        assert "search" in called_url

    def test_passes_query_param(self):
        mock_resp = _mock_response(features=[_make_feature()])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp) as mock_get:
            geocoding_search.invoke({"q": "ma recherche"})
        params = mock_get.call_args[1]["params"] if "params" in mock_get.call_args[1] else mock_get.call_args[0][1]
        assert params.get("q") == "ma recherche"

    def test_limit_parameter_passed(self):
        mock_resp = _mock_response(features=[_make_feature()])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp) as mock_get:
            geocoding_search.invoke({"q": "Paris", "limit": 3})
        params = mock_get.call_args[1]["params"] if "params" in mock_get.call_args[1] else mock_get.call_args[0][1]
        assert params.get("limit") == 3

    def test_score_present_in_result(self):
        mock_resp = _mock_response(features=[_make_feature(score=0.87)])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = geocoding_search.invoke({"q": "Paris"})
        assert "0.87" in result


# ---------------------------------------------------------------------------
# reverse_geocoding
# ---------------------------------------------------------------------------

class TestReverseGeocoding:
    def test_returns_string(self):
        mock_resp = _mock_response(features=[_make_feature(label="1 Rue de Rivoli, Paris")])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = reverse_geocoding.invoke({"lon": 2.3522, "lat": 48.8566})
        assert isinstance(result, str)

    def test_contains_address_in_result(self):
        label = "1 Rue de Rivoli, Paris"
        mock_resp = _mock_response(features=[_make_feature(label=label)])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = reverse_geocoding.invoke({"lon": 2.3522, "lat": 48.8566})
        assert label in result

    def test_no_results_returns_aucune_adresse(self):
        mock_resp = _mock_response(features=[])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp):
            result = reverse_geocoding.invoke({"lon": 0.0, "lat": 0.0})
        assert "Aucune adresse" in result

    def test_http_error_returns_error_message(self):
        import requests as req
        with patch("src.agentia.geocoding_tool.requests.get",
                   side_effect=req.exceptions.Timeout("Timeout")):
            result = reverse_geocoding.invoke({"lon": 2.3522, "lat": 48.8566})
        assert isinstance(result, str)

    def test_calls_reverse_endpoint(self):
        mock_resp = _mock_response(features=[_make_feature()])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp) as mock_get:
            reverse_geocoding.invoke({"lon": 2.3522, "lat": 48.8566})
        called_url = mock_get.call_args[0][0]
        assert "reverse" in called_url

    def test_passes_lon_lat_params(self):
        mock_resp = _mock_response(features=[_make_feature()])
        with patch("src.agentia.geocoding_tool.requests.get", return_value=mock_resp) as mock_get:
            reverse_geocoding.invoke({"lon": 4.8357, "lat": 45.7640})
        params = mock_get.call_args[1]["params"] if "params" in mock_get.call_args[1] else mock_get.call_args[0][1]
        assert params.get("lon") == 4.8357
        assert params.get("lat") == 45.7640
