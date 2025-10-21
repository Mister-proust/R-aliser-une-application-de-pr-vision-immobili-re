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
    payload = {"surface": 45, "rooms": 2}
    r = client.post("/predict", json=payload)
    # should be unauthorized when no Authorization header provided
    assert r.status_code == 401
    j = r.json()
    assert "detail" in j


def test_predict_with_token(client):
    from src.app import config

    payload = {"surface": 45, "rooms": 2}
    headers = {"Authorization": f"Bearer {config.API_TOKEN}"}
    r = client.post("/predict", json=payload, headers=headers)
    assert r.status_code == 200
    j = r.json()
    assert "estimated_price" in j
    # expected heuristic: surface*2500 + rooms*5000
    assert j["estimated_price"] == round(45 * 2500 + 2 * 5000, 2)


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
    # estimated price should be numeric (0.0 with current heuristic)
    assert "estimated_price" in j
    assert isinstance(j["estimated_price"], (int, float))
