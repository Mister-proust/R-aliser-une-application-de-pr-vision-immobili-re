# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import fastapi
import gradio

load_dotenv()
fastapi_port = os.getenv("FASTAPI_PORT", "8000")
API_TITLE = "Prédiction de valeures immobilières à partir des données DVF"
API_DESCRIPTION = """
API permettant de prédire les valeurs immobilières en utilisant un modèle de machine learning entraîné sur les données DVF (Demandes de Valeurs Foncières).
    """
HOST = "0.0.0.0"
PORT = int(fastapi_port)
RELOAD = True
API_TOKEN = os.getenv("API_TOKEN", "secret-token")
MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "models" / "xgb_pipeline.pkl"
gradio_port = os.getenv("GRADIO_PORT", "7860")
gradio_host = os.getenv("GRADIO_HOST", "127.0.0.1")