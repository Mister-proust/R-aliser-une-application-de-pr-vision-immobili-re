# config.py
import os
from pathlib import Path
# API configuration
API_TITLE = "Prédiction de valeures immobilières à partir des données DVF"
API_DESCRIPTION = """
API permettant de prédire les valeurs immobilières en utilisant un modèle de machine learning entraîné sur les données DVF (Demandes de Valeurs Foncières).
    """

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True
# Simple API token for a lightweight auth. Can be overridden with the API_TOKEN env var.
API_TOKEN = os.getenv("API_TOKEN", "secret-token")
MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "xgb_pipeline.pkl"