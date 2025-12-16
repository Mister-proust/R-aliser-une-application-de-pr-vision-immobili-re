import pytest
from fastapi.testclient import TestClient
import os
import sys

# remonter d'un niveau (ajouter le répertoire parent au PYTHONPATH)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_requires_token(client):
    # new API expects property-level fields; token still required
    payload = {"commune": "Paris", "type_voie": "Rue", "type_local": "Appartement", "surface_reelle_bati": 45, "nombre_pieces_principales": 2}
    r = client.post("/predict", json=payload)
    # should be unauthorized when no Authorization header provided
    assert r.status_code == 401
    j = r.json()
    assert "detail" in j


def test_predict_with_token(client):
    from src.app import config
    # send property-level payload; pipeline may exist or fallback will be returned
    payload = {"commune": "Paris", "type_voie": "Rue", "type_local": "Appartement", "surface_reelle_bati": 45, "nombre_pieces_principales": 2}
    headers = {"Authorization": f"Bearer {config.API_TOKEN}"}
    r = client.post("/predict", json=payload, headers=headers)
    assert r.status_code == 200
    j = r.json()
    # either pipeline returned top-level estimated_price, or a fallback result is included
    assert ("estimated_price" in j and isinstance(j["estimated_price"], (int, float))) or (
        "result" in j and isinstance(j["result"].get("estimated_price", None), (int, float))
    )


def test_predict_with_wrong_token(client):
    payload = {"surface": 10, "rooms": 1}
    headers = {"Authorization": "Bearer wrong-token"}
    r = client.post("/predict", json=payload, headers=headers)
    assert r.status_code == 401
    assert "detail" in r.json()


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    # Should be HTML content
    assert "text/html" in r.headers.get("content-type", "")
    # Contains some text from the template
    assert b"Formulaire d'estimation" in r.content or b"Estimation DVF" in r.content


def test_predict_with_token_and_missing_fields(client):
    from src.app import config

    # send an empty payload but with valid token
    headers = {"Authorization": f"Bearer {config.API_TOKEN}"}
    r = client.post("/predict", json={}, headers=headers)
    assert r.status_code == 200
    j = r.json()
    # estimated price should be numeric either at top-level or inside result (fallback)
    assert ("estimated_price" in j and isinstance(j["estimated_price"], (int, float))) or (
        "result" in j and isinstance(j["result"].get("estimated_price", None), (int, float))
    )


def test_meta_options(client):
    r = client.get('/meta/options')
    assert r.status_code == 200
    j = r.json()
    assert 'type_voie' in j and isinstance(j['type_voie'], list)
    assert 'type_local' in j and isinstance(j['type_local'], list)


def test_pipeline_file_exists():
    from src.app import config
    import os
    path = str(config.MODEL_PATH)
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"Pipeline file not found at {path}")
    assert os.path.exists(path)
